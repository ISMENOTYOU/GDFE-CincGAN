import os
import json
import warnings
from contextlib import contextmanager

import numpy as np

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from .utils import get_lr
from .utils_metrics import PSNR, SSIM


@contextmanager
def temporary_eval(*models, freeze=False):
    """Restore both module modes and original requires_grad flags on exit."""
    modes = [(module, module.training) for model in models for module in model.modules()]
    parameters = [(p, p.requires_grad) for model in models for p in model.parameters()]
    try:
        for model in models:
            model.eval()
        if freeze:
            for p, _ in parameters:
                p.requires_grad_(False)
        yield
    finally:
        for module, training in modes:
            module.training = training
        for p, requires_grad in parameters:
            p.requires_grad_(requires_grad)


def require_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('{} contains NaN or Inf; no optimizer update performed for this loss.'.format(name))


def optimizer_step(loss, optimizer, model, name, max_norm):
    require_finite(name, loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    # Check the aggregate gradient norm BEFORE changing any parameter.
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()


def tensor_stats(tensor):
    tensor = tensor.detach().float()
    require_finite('preview output', tensor)
    return dict(min=tensor.min().item(), max=tensor.max().item(),
                mean=tensor.mean().item(), std=tensor.std(unbiased=False).item(),
                saturated_fraction=(tensor.abs() >= 0.99).float().mean().item())


def save_epoch_preview(G_model, G1_model, dataset, epoch, save_dir):
    """Save the same training sample after each epoch using updated weights."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # This dataset has no random augmentation, so index 0 is comparable across
    # epochs. Access it directly without consuming or reshuffling the loader.
    lr_image, nlr_image, hr_image = dataset[0]
    parameter = next(G1_model.parameters())
    lr_input = torch.as_tensor(
        lr_image, device=parameter.device, dtype=parameter.dtype,
    ).unsqueeze(0)

    # Use the unwrapped models on rank 0 to avoid DDP forward collectives.
    modes = [(module, module.training)
             for model in (G1_model, G_model) for module in model.modules()]
    try:
        G1_model.eval()
        G_model.eval()
        with torch.no_grad():
            denoised = G1_model(lr_input)
            sr_image = G_model(denoised)
    finally:
        for module, training in modes:
            module.training = training

    images = (lr_image, nlr_image, denoised[0], sr_image[0], hr_image)
    titles = ('LR input', 'NLR reference', 'G1 denoised', 'G super-resolution', 'HR reference')
    output_dir = os.path.join(save_dir, 'epoch_images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'epoch_{:03d}.png'.format(epoch))
    stats = dict(epoch=epoch, sample_index=0, G1=tensor_stats(denoised), G=tensor_stats(sr_image))
    with torch.no_grad():
        nlr_target = torch.as_tensor(nlr_image, device=denoised.device).unsqueeze(0)
        hr_target = torch.as_tensor(hr_image, device=sr_image.device).unsqueeze(0)
        stats['G1']['mse'] = F.mse_loss(denoised, nlr_target).item()
        stats['G']['mse'] = F.mse_loss(sr_image, hr_target).item()
        stats['G']['psnr'] = PSNR(sr_image, hr_target).item()
        stats['G']['ssim'] = SSIM(sr_image, hr_target).item()
    # Raw tensors distinguish saturation from plotting/normalization problems.
    arrays = [torch.as_tensor(value).detach().float().cpu().numpy() for value in images]
    np.savez_compressed(os.path.splitext(output_path)[0] + '.npz',
                        **dict(zip(('lr', 'nlr', 'denoised', 'sr', 'hr'), arrays)))
    with open(os.path.splitext(output_path)[0] + '.json', 'w', encoding='utf-8') as stream:
        json.dump(stats, stream, indent=2)
    for name in ('G1', 'G'):
        if stats[name]['std'] < 1e-4 or stats[name]['saturated_fraction'] > 0.98:
            warnings.warn('Epoch {} {} preview is almost constant or saturated: {}'.format(
                epoch, name, stats[name]), RuntimeWarning)
    fig = Figure(figsize=(15, 3.6))
    FigureCanvasAgg(fig)
    try:
        for ax, image, title in zip(fig.subplots(1, 5), images, titles):
            image = torch.as_tensor(image).detach().float().cpu().numpy()
            ax.imshow(image[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle('Epoch {} | training sample 0 | normalized range [-1, 1]'.format(epoch))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output_path, dpi=150)
    finally:
        fig.clear()
    return output_path


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def fit_one_epoch(G_model_train, G1_model_train, G2_model_train, F_model_train, D_model_train, D2_model_train,
                  G_model, G1_model, G2_model, F_model, D_model, D2_model,
                  VGG_feature_model, loss_history, G_optimizer, G1_optimizer, G2_optimizer,
                  F_optimizer, D_optimizer, D2_optimizer, BCE_loss, MSE_loss,
                  epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir,
                  local_rank=0):
    if fp16:
        raise ValueError('This six-network training loop requires fp16=False.')
    if torch.distributed.is_initialized():
        raise ValueError('Use distributed=False (single GPU or DataParallel) for this loop.')
    max_norm = float(os.environ.get('SRGAN_GRAD_CLIP', '1.0'))
    warmup = int(os.environ.get('SRGAN_WARMUP_EPOCHS', '10'))
    ramp_epochs = max(1, int(os.environ.get('SRGAN_RAMP_EPOCHS', '10')))
    cycle_g1 = float(os.environ.get('SRGAN_CYCLE_G1', '1.0'))
    cycle_g = float(os.environ.get('SRGAN_CYCLE_G', '0.1'))
    if max_norm <= 0 or warmup < 0 or cycle_g1 < 0 or cycle_g < 0:
        raise ValueError('Invalid gradient clipping, warmup or cycle weight.')
    # A supervised warmup is a new, configurable experiment setting.
    auxiliary_scale = min(1.0, max(0.0, (epoch - warmup + 1) / ramp_epochs))
    adv_weight = 1e-3 * auxiliary_scale
    totals = {}
    processed = 0
    pbar = tqdm(total=epoch_step, desc='Epoch {}/{}'.format(epoch + 1, Epoch),
                mininterval=0.3, disable=local_rank != 0)
    if local_rank == 0:
        print('Start Train | cycle/adversarial ramp={:.3f}'.format(auxiliary_scale))
    try:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            device = next(G_model.parameters()).device
            lr_images, nlr_images, hr_images = [value.to(device, non_blocking=True) for value in batch]
            for name, value in zip(('LR', 'NLR', 'HR'), (lr_images, nlr_images, hr_images)):
                require_finite('{} epoch {} batch {}'.format(name, epoch + 1, iteration), value)
            batch_size = lr_images.shape[0]
            y_real = torch.ones(batch_size, device=device)
            y_fake = torch.zeros(batch_size, device=device)
            zero = lr_images.new_zeros(())

            # D2 sees detached fakes in eval mode: no extra G1 BatchNorm updates.
            D2_train_loss = zero
            if adv_weight > 0:
                with temporary_eval(G1_model_train), torch.no_grad():
                    fake_nlr = G1_model_train(lr_images)
                D2_train_loss = (BCE_loss(D2_model_train(nlr_images).view(-1), y_real)
                                 + BCE_loss(D2_model_train(fake_nlr).view(-1), y_fake))
                optimizer_step(D2_train_loss, D2_optimizer, D2_model_train, 'D2 loss', max_norm)

            # Train G2 on a detached input, then rebuild G1's cycle graph using
            # the updated, frozen G2. No shared graph survives an optimizer step.
            G1_result = G1_model_train(lr_images)
            require_finite('G1 output', G1_result)
            G2_train_loss = MSE_loss(G2_model_train(G1_result.detach()), lr_images)
            optimizer_step(G2_train_loss, G2_optimizer, G2_model_train, 'G2 loss', max_norm)
            G1_image_loss = MSE_loss(G1_result, nlr_images)
            G2_G1_loss = zero
            G12_adversarial_loss = zero
            if auxiliary_scale > 0:
                with temporary_eval(G2_model_train, freeze=True):
                    G2_G1_loss = MSE_loss(G2_model_train(G1_result), lr_images)
                with temporary_eval(D2_model_train, freeze=True):
                    G12_adversarial_loss = BCE_loss(D2_model_train(G1_result).view(-1), y_real)
            G1_train_loss = (G1_image_loss + auxiliary_scale * cycle_g1 * G2_G1_loss
                             + adv_weight * G12_adversarial_loss)
            optimizer_step(G1_train_loss, G1_optimizer, G1_model_train, 'G1 loss', max_norm)

            # Feed G with the same G1 inference mode used by deployment/preview.
            with temporary_eval(G1_model_train), torch.no_grad():
                denoised = G1_model_train(lr_images)
            require_finite('G1 eval output', denoised)
            D_train_loss = zero
            if adv_weight > 0:
                with temporary_eval(G_model_train), torch.no_grad():
                    fake_hr = G_model_train(denoised)
                D_train_loss = (BCE_loss(D_model_train(hr_images).view(-1), y_real)
                                + BCE_loss(D_model_train(fake_hr).view(-1), y_fake))
                optimizer_step(D_train_loss, D_optimizer, D_model_train, 'D loss', max_norm)

            G_result = G_model_train(denoised)
            require_finite('G output', G_result)
            F_train_loss = MSE_loss(F_model_train(G_result.detach()), lr_images)
            optimizer_step(F_train_loss, F_optimizer, F_model_train, 'F loss', max_norm)
            G_image_loss = MSE_loss(G_result, hr_images)
            F_G_loss = zero
            GF_adversarial_loss = zero
            if auxiliary_scale > 0:
                with temporary_eval(F_model_train, freeze=True):
                    F_G_loss = MSE_loss(F_model_train(G_result), lr_images)
                with temporary_eval(D_model_train, freeze=True):
                    GF_adversarial_loss = BCE_loss(D_model_train(G_result).view(-1), y_real)
            G_train_loss = G_image_loss + auxiliary_scale * cycle_g * F_G_loss + adv_weight * GF_adversarial_loss
            optimizer_step(G_train_loss, G_optimizer, G_model_train, 'G loss', max_norm)

            with torch.no_grad():
                values = dict(G_total_loss=G_train_loss.item(), G1_total_loss=G1_train_loss.item(),
                              D_total_loss=D_train_loss.item(), D2_total_loss=D2_train_loss.item(),
                              G_image_loss=G_image_loss.item(), G1_image_loss=G1_image_loss.item(),
                              cycle_G1_loss=G2_G1_loss.item(), cycle_G_loss=F_G_loss.item(),
                              G_total_PSNR=PSNR(G_result, hr_images).item(),
                              G_total_SSIM=SSIM(G_result, hr_images).item(),
                              G1_total_PSNR=PSNR(G1_result, nlr_images).item(),
                              G1_total_SSIM=SSIM(G1_result, nlr_images).item(),
                              G_saturated_fraction=(G_result.abs() >= .99).float().mean().item(),
                              G1_saturated_fraction=(G1_result.abs() >= .99).float().mean().item())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            processed += 1
            pbar.set_postfix(G_loss=totals['G_total_loss'] / processed,
                             G1_loss=totals['G1_total_loss'] / processed,
                             PSNR=totals['G_total_PSNR'] / processed,
                             lr=get_lr(G_optimizer))
            pbar.update(1)
    finally:
        pbar.close()
    if processed == 0:
        raise ValueError('No complete training batches were produced.')
    means = {key: value / processed for key, value in totals.items()}
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print('Epoch: {}/{} | G loss: {:.6f} | G1 loss: {:.6f}'.format(
            epoch + 1, Epoch, means['G_total_loss'], means['G1_total_loss']))
        models = dict(G=G_model, G1=G1_model, G2=G2_model, F=F_model, D=D_model, D2=D2_model)
        optimizers = dict(G=G_optimizer, G1=G1_optimizer, G2=G2_optimizer,
                          F=F_optimizer, D=D_optimizer, D2=D2_optimizer)
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            for name in ('G', 'G1'):
                filename = '{}_Epoch{}-GLoss{:.4f}-G1Loss{:.4f}.pth'.format(
                    name, epoch + 1, means['G_total_loss'], means['G1_total_loss'])
                torch.save(models[name].state_dict(), os.path.join(save_dir, filename))
        for name in ('G', 'G1'):
            torch.save(models[name].state_dict(), os.path.join(save_dir, name + '_model_last_epoch_weights.pth'))
        checkpoint = dict(epoch=epoch + 1, models={name: model.state_dict() for name, model in models.items()},
                          optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
                          training_options=dict(max_norm=max_norm, warmup=warmup, ramp_epochs=ramp_epochs,
                                                cycle_g1=cycle_g1, cycle_g=cycle_g), metrics=means)
        checkpoint['schedule'] = getattr(G_model, '_training_schedule', {})
        checkpoint_path = os.path.join(save_dir, 'training_last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path + '.tmp')
        os.replace(checkpoint_path + '.tmp', checkpoint_path)
        preview_path = save_epoch_preview(G_model, G1_model, gen.dataset, epoch + 1, save_dir)
        with open(os.path.splitext(preview_path)[0] + '.json', encoding='utf-8') as stream:
            preview = json.load(stream)
        # These are fixed TRAINING sample diagnostics, not validation-set scores.
        means.update(sample_G1_eval_mse=preview['G1']['mse'], sample_G_eval_mse=preview['G']['mse'],
                     sample_G_eval_PSNR=preview['G']['psnr'], sample_G_eval_SSIM=preview['G']['ssim'])
        loss_history.append_loss(epoch + 1, **means)
        print('Preview saved: {} | G1 std={:.6g}, saturation={:.2%} | G std={:.6g}, saturation={:.2%}'.format(
            preview_path, preview['G1']['std'], preview['G1']['saturated_fraction'],
            preview['G']['std'], preview['G']['saturated_fraction']))
