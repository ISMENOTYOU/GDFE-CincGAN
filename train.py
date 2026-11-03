import datetime
import os
import warnings

# Set visibility before importing PyTorch. A launch-time setting takes priority.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader
from nets.srgan import Discriminator, Generatorup, Generatordown, Generator
from utils.callbacks import LossHistory
from utils.dataloader import SRGAN_dataset_collate, SRGANDataset
from utils.utils import (get_lr_scheduler, set_optimizer_lr,
                         show_config, set_optimizer_lr2)
from utils.utils_fit1 import fit_one_epoch


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # 初始化函数
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   当前稳定性训练循环支持单卡和DataParallel，尚未支持DDP。
    # ---------------------------------------------------------------------#
    distributed = False
    if distributed:
        raise ValueError('This stability experiment supports single GPU / DataParallel; use distributed=False.')
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    if fp16:
        raise ValueError('The six-network stability experiment currently requires fp16=False.')
    # ---------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    # ---------------------------------------------------------------------#
    G_model_path = ""
    G1_model_path = ""
    G2_model_path = ""
    F_model_path = ""
    D_model_path = ""
    D2_model_path = ""
    # -----------------------------------#
    #   代表进行四倍的上采样
    # -----------------------------------#
    scale_factor = 2
    # -----------------------------------#
    #   获得输入与输出的图片的shape
    # -----------------------------------#
    lr_shape = [128, 128]
    nlr_shape = [128, 128]
    hr_shape = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]

    # -----------------------------------------------------#
    #   训练参数设置
    #   Init_epoch      显示的起始世代，默认为0
    #                   断点续练时可调整，会自动调整学习率
    #   Epoch           总共训练的Epoch
    #   batch_size      DP模式下为两张卡合计的批次大小，不是每张卡的大小
    # -----------------------------------------------------#
    Init_Epoch = 0
    Epoch = int(os.environ.get('SRGAN_EPOCHS', '300'))
    # Default to the user's single-GPU run; set 14 explicitly for two GPUs.
    batch_size = int(os.environ.get("SRGAN_BATCH_SIZE", "7"))
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    # Conservative NEW experiment setting, not proof of the old failure cause.
    Init_lr = float(os.environ.get("SRGAN_INIT_LR", "5e-5"))
    Min_lr = Init_lr * 0.01
    Init_lr2 = float(os.environ.get("SRGAN_D_INIT_LR", "1e-5"))
    Min_lr2 = Init_lr2 * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=2e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.5
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = os.environ.get("SRGAN_SAVE_DIR", "logs")
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = int(os.environ.get('SRGAN_NUM_WORKERS', '2'))
    # ------------------------------#
    #   每隔50个step保存一次图片
    # ------------------------------#
    # photo_save_step = 40
    # ------------------------------#
    #   获得图片路径
    # ------------------------------#
    dataset_path = os.environ.get('SRGAN_DATASET_DIR', './Unpaired henan')
    annotation_path_HR = os.path.join(dataset_path, 'train_lines_HR_npy.txt')
    annotation_path_LR = os.path.join(dataset_path, 'train_lines_LR_npy.txt')
    annotation_path_NLR = os.path.join(dataset_path, 'train_lines_NLR_npy.txt')
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    if Cuda and not torch.cuda.is_available():
        raise RuntimeError("Cuda=True, but CUDA is unavailable in this Python environment.")
    ngpus_per_node = torch.cuda.device_count()
    if Cuda and not distributed:
        if batch_size < 2 * ngpus_per_node or batch_size % ngpus_per_node != 0:
            raise ValueError(
                "DataParallel requires an evenly divisible global batch with at least "
                "2 samples per GPU. GPU count={}, batch_size={}".format(ngpus_per_node, batch_size)
            )
        torch.cuda.set_device(0)
        print("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("DataParallel: {} GPU(s), total batch={}, batch per GPU={}".format(
            ngpus_per_node, batch_size, batch_size // ngpus_per_node))
        for gpu_index in range(ngpus_per_node):
            print("GPU {}: {}".format(gpu_index, torch.cuda.get_device_name(gpu_index)))
    if distributed:
        dist.init_process_group(backend="ncc")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # ---------------------------#
    #   生成网络和评价网络
    # ---------------------------#
    G_model = Generatorup(scale_factor)
    G1_model = Generator()
    G2_model = Generator()
    F_model = Generatordown()
    init_weights(G_model)
    init_weights(G1_model)
    init_weights(G2_model)
    init_weights(F_model)

    D_model = Discriminator()
    D2_model = Discriminator()
    init_weights(D_model)
    init_weights(D2_model)

    # Only new full checkpoints support continuation of all six networks.
    resume_checkpoint = None
    resume_path = os.environ.get("SRGAN_RESUME", "").strip()
    if resume_path:
        if any((G_model_path, G1_model_path, G2_model_path, F_model_path, D_model_path, D2_model_path)):
            raise ValueError('Use SRGAN_RESUME OR individual model paths, not both.')
        resume_checkpoint = torch.load(resume_path, map_location='cpu')
        for name, model in dict(G=G_model, G1=G1_model, G2=G2_model,
                                F=F_model, D=D_model, D2=D2_model).items():
            model.load_state_dict(resume_checkpoint['models'][name], strict=True)
        Init_Epoch = int(resume_checkpoint['epoch'])
        for name, env_name in dict(max_norm='SRGAN_GRAD_CLIP', warmup='SRGAN_WARMUP_EPOCHS',
                                   ramp_epochs='SRGAN_RAMP_EPOCHS', cycle_g1='SRGAN_CYCLE_G1',
                                   cycle_g='SRGAN_CYCLE_G').items():
            os.environ.setdefault(env_name, str(resume_checkpoint['training_options'][name]))
        print('Resuming six networks after completed epoch {}'.format(Init_Epoch))

    # -----------------------------------#
    #   创建VGG模型，该模型用于提取特征
    # -----------------------------------#
    # Perceptual loss is inactive; avoid downloading/allocating an unused RGB VGG.
    VGG_feature_model = None

    # ------------------------------------------#
    #   将训练好的模型重新载入（将预训练的权重加载到生成器和判别器模型中，并确保权重的维度匹配。这在迁移学习或继续训练时很常见，以利用预训练模型在新任务上的特征学习）
    # ------------------------------------------#
    if G_model_path != '':
        model_dict = G_model.state_dict()
        pretrained_dict = torch.load(G_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        G_model.load_state_dict(model_dict)
    if G1_model_path != '':
        model_dict = G1_model.state_dict()
        pretrained_dict = torch.load(G1_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        G1_model.load_state_dict(model_dict)
    if G2_model_path != '':
        model_dict = G2_model.state_dict()
        pretrained_dict = torch.load(G2_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        G2_model.load_state_dict(model_dict)
    if F_model_path != '':
        model_dict = F_model.state_dict()
        pretrained_dict = torch.load(F_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        F_model.load_state_dict(model_dict)
    if D_model_path != '':
        model_dict = D_model.state_dict()
        pretrained_dict = torch.load(D_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        D_model.load_state_dict(model_dict)
    if D2_model_path != '':
        model_dict = D2_model.state_dict()
        pretrained_dict = torch.load(D2_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        D2_model.load_state_dict(model_dict)

    # ----------------------#
    #   获得损失函数
    # ----------------------#
    BCE_loss = nn.BCEWithLogitsLoss()
    MSE_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        graph_models = [G_model] if os.environ.get('SRGAN_LOG_GRAPH', '0') == '1' else []
        loss_history = LossHistory(log_dir, graph_models, input_shape=lr_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    G_model_train = G_model.train()
    G1_model_train = G1_model.train()
    G2_model_train = G2_model.train()
    F_model_train = F_model.train()
    D_model_train = D_model.train()
    D2_model_train = D2_model.train()

    if Cuda:
        if distributed:
            G_model_train = G_model_train.cuda(local_rank)
            G_model_train = torch.nn.parallel.DistributedDataParallel(G_model_train, device_ids=[local_rank],
                                                                      find_unused_parameters=True)

            G1_model_train = G1_model_train.cuda(local_rank)
            G1_model_train = torch.nn.parallel.DistributedDataParallel(G1_model_train, device_ids=[local_rank],
                                                                       find_unused_parameters=True)

            G2_model_train = G2_model_train.cuda(local_rank)
            G2_model_train = torch.nn.parallel.DistributedDataParallel(G2_model_train, device_ids=[local_rank],
                                                                       find_unused_parameters=True)

            F_model_train = F_model_train.cuda(local_rank)
            F_model_train = torch.nn.parallel.DistributedDataParallel(F_model_train, device_ids=[local_rank],
                                                                      find_unused_parameters=True)

            D_model_train = D_model_train.cuda(local_rank)
            D_model_train = torch.nn.parallel.DistributedDataParallel(D_model_train, device_ids=[local_rank],
                                                                      find_unused_parameters=True)
            D2_model_train = D2_model_train.cuda(local_rank)
            D2_model_train = torch.nn.parallel.DistributedDataParallel(D2_model_train, device_ids=[local_rank],
                                                                       find_unused_parameters=True)

        else:
            cudnn.benchmark = True
            gpu_ids = list(range(ngpus_per_node))
            G_model_train = torch.nn.DataParallel(G_model, device_ids=gpu_ids)
            G_model_train = G_model_train.cuda()

            G1_model_train = torch.nn.DataParallel(G1_model, device_ids=gpu_ids)
            G1_model_train = G1_model_train.cuda()

            G2_model_train = torch.nn.DataParallel(G2_model, device_ids=gpu_ids)
            G2_model_train = G2_model_train.cuda()

            F_model_train = torch.nn.DataParallel(F_model, device_ids=gpu_ids)
            F_model_train = F_model_train.cuda()

            D_model_train = torch.nn.DataParallel(D_model, device_ids=gpu_ids)
            D_model_train = D_model_train.cuda()

            D2_model_train = torch.nn.DataParallel(D2_model, device_ids=gpu_ids)
            D2_model_train = D2_model_train.cuda()


    with open(annotation_path_HR) as f:
        lines_HR = f.readlines()
    with open(annotation_path_LR) as f:
        lines_LR = f.readlines()
    with open(annotation_path_NLR) as f:
        lines_NLR = f.readlines()
    num_train = len(lines_HR)

    if local_rank == 0:
        show_config(
            lr_shape=lr_shape, nlr_shape=nlr_shape, hr_shape=hr_shape, Init_Epoch=Init_Epoch, Epoch=Epoch,
            batch_size=batch_size,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train
        )

    # ------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    # ------------------------------------------------------#
    if True:
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        G_optimizer = {
            'adam': optim.Adam(G_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(G_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        G1_optimizer = {
            'adam': optim.Adam(G1_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(G1_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        G2_optimizer = {
            'adam': optim.Adam(G2_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(G2_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        F_optimizer = {
            'adam': optim.Adam(F_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(F_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        D_optimizer = {
            'adam': optim.Adam(D_model_train.parameters(), lr=Init_lr2, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(D_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        D2_optimizer = {
            'adam': optim.Adam(D2_model_train.parameters(), lr=Init_lr2, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(D2_model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        if resume_checkpoint is not None:
            for name, optimizer in dict(G=G_optimizer, G1=G1_optimizer, G2=G2_optimizer,
                                        F=F_optimizer, D=D_optimizer, D2=D2_optimizer).items():
                optimizer.load_state_dict(resume_checkpoint['optimizers'][name])
            # Preserve the original scheduler for new-format checkpoints.
            saved = resume_checkpoint.get('schedule', {})
            Init_lr = saved.get('Init_lr', Init_lr)
            Min_lr = saved.get('Min_lr', Min_lr)
            Init_lr2 = saved.get('Init_lr2', Init_lr2)
            Min_lr2 = saved.get('Min_lr2', Min_lr2)
            Epoch = saved.get('Epoch', Epoch)

        # Store scheduler metadata on the raw model for the epoch checkpoint.
        G_model._training_schedule = dict(Init_lr=Init_lr, Min_lr=Min_lr,
                                         Init_lr2=Init_lr2, Min_lr2=Min_lr2, Epoch=Epoch)

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        lr_scheduler_func2 = get_lr_scheduler(lr_decay_type, Init_lr2, Min_lr2, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        train_dataset = SRGANDataset(lines_LR, lines_NLR, lines_HR, dataset_path, lr_shape, nlr_shape, hr_shape)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=SRGAN_dataset_collate, sampler=train_sampler)

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(G_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(G1_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(G2_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(F_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr2(D_optimizer, lr_scheduler_func2, epoch)
            set_optimizer_lr2(D2_optimizer, lr_scheduler_func2, epoch)

            fit_one_epoch(G_model_train, G1_model_train, G2_model_train, F_model_train, D_model_train, D2_model_train,
                          G_model, G1_model, G2_model, F_model, D_model,
                          D2_model, VGG_feature_model, loss_history, G_optimizer, G1_optimizer, G2_optimizer,
                          F_optimizer, D_optimizer, D2_optimizer, BCE_loss, MSE_loss,
                          epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()
        # 开始模型训练
        # for epoch in range(Init_Epoch, Epoch):
        #
        #     if distributed:
        #         train_sampler.set_epoch(epoch)
        #
        #     set_optimizer_lr(G1_optimizer, lr_scheduler_func, epoch)
        #     set_optimizer_lr(G2_optimizer, lr_scheduler_func, epoch)
        #
        #     # 判别器只在每两个epoch训练一次
        #     if epoch % 2 == 0:
        #         set_optimizer_lr2(D_optimizer, lr_scheduler_func2, epoch)
        #         set_optimizer_lr2(D2_optimizer, lr_scheduler_func2, epoch)
        #
        #         fit_one_epoch(
        #             G_model_train, G1_model_train, G2_model_train, F_model_train, D_model_train, D2_model_train,
        #             G_model, G1_model, G2_model, F_model, D_model, D2_model, VGG_feature_model, loss_history,
        #             G_optimizer, G1_optimizer, G2_optimizer, F_optimizer, D_optimizer, D2_optimizer,
        #             BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir,
        #             local_rank=local_rank
        #         )
        #     else:
        #         fit_one_epoch(
        #             G_model_train, G1_model_train, G2_model_train, F_model_train, None, None,  # 不训练判别器
        #             G_model, G1_model, G2_model, F_model, None, None, VGG_feature_model, loss_history,
        #             G_optimizer, G1_optimizer, G2_optimizer, F_optimizer, None, None,  # 判别器优化器为None
        #             BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir,
        #             local_rank=local_rank
        #         )
