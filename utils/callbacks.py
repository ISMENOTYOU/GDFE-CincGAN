import os
import warnings

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        for m in model:
            modes = [(module, module.training) for module in m.modules()]
            try:
                parameter = next(m.parameters())
                dummy_input = torch.randn(
                    2, 1, input_shape[0], input_shape[1],
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                # Graph logging must not update BatchNorm training statistics.
                m.eval()
                with torch.no_grad():
                    self.writer.add_graph(m, dummy_input)
            except Exception as error:
                warnings.warn('TensorBoard graph logging skipped: {}'.format(error), RuntimeWarning)
            finally:
                for module, training in modes:
                    module.training = training

    def append_loss(self, epoch, **kwargs):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not hasattr(self, 'epochs'):
            self.epochs = []
        self.epochs.append(epoch)
        with open(os.path.join(self.log_dir, 'epoch_numbers.txt'), 'a') as stream:
            stream.write(str(epoch) + '\n')

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, [])
            # ---------------------------------#
            #   为列表添加数值
            # ---------------------------------#
            getattr(self, key).append(value)

            # ---------------------------------#
            #   写入txt
            # ---------------------------------#
            with open(os.path.join(self.log_dir, key + ".txt"), 'a') as f:
                f.write(str(value))
                f.write("\n")

            # ---------------------------------#
            #   写入tensorboard
            # ---------------------------------#
            self.writer.add_scalar(key, value, epoch)

        self.loss_plot(**kwargs)

    def loss_plot(self, **kwargs):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for key in kwargs:
            if 'PSNR' in key:
                index = 1
            elif 'SSIM' in key:
                index = 2
            elif 'saturated_fraction' in key:
                index = 3
            else:
                index = 0
            values = getattr(self, key)
            axes.flat[index].plot(self.epochs[-len(values):], values, linewidth=1.3, label=key)
        for ax, title in zip(axes.flat, ('Loss / MSE', 'PSNR (dB)', 'SSIM', 'Saturated output fraction')):
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.grid(True)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'epoch_loss.png'), dpi=130)
        plt.close(fig)
