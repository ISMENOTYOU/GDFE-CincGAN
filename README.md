# 地震数据去噪与二倍超分辨率重建
## 1. 模型与训练流程

主要推理路径：

```text
LR：带噪低分辨率输入，128×128
                │
                ▼
          G1：去噪网络
                │
                ▼
      去噪后的低分辨率数据
                │
                ▼
        G：二倍超分辨率网络
                │
                ▼
       SR：256×256重建结果
```

| 网络 | 作用 |
|---|---|
| G1 | 将LR转换为接近NLR的低分辨率数据 |
| G2 | 将G1结果重建回LR，提供逆映射和循环约束 |
| D2 | 区分真实NLR与G1结果 |
| G | 将去噪结果放大为二倍超分辨率结果 |
| F | 将SR重建回LR，提供逆映射和循环约束 |
| D | 区分真实HR与G结果 |


## 2. 目录结构

```text
.
├── train.py
├── run_supervised_baseline.sh
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
├── nets/
│   ├── __init__.py
│   └── srgan.py
├── utils/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── utils_fit1.py
│   ├── callbacks.py
│   ├── utils_metrics.py
│   └── utils.py
├── Unpaired henan/
│   ├── LR_npy/
│   ├── NLR_npy/
│   ├── HR_npy/
│   ├── train_lines_LR_npy.txt
│   ├── train_lines_NLR_npy.txt
│   └── train_lines_HR_npy.txt
├── COPY_MANIFEST.csv
├── PACKAGE_SUMMARY.json
└── DATA_CHECK_REPORT.json
```


## 3. 运行环境

优先使用原来已经能正常运行的Python/PyTorch环境。新建环境时，先安装与机器及显卡驱动匹配的PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

参见[PyTorch安装说明](https://pytorch.org/get-started/locally/)。依赖文件是版本范围，不是完整锁定的复现实验环境。

当前入口默认 `Cuda=True`，需要可用CUDA环境。代码支持单卡与 `DataParallel`；`distributed=True` 或 `fp16=True` 会明确报错，当前未提供DDP和混合精度实现。VGG感知损失未参与当前损失，因此代码不加载VGG，也不需要下载其预训练权重。

## 4. 默认300轮训练

进入仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 SRGAN_BATCH_SIZE=7 SRGAN_DATASET_DIR="./Unpaired henan" SRGAN_SAVE_DIR=logs_full_experiment python train.py
```

默认配置：

| 参数 | 默认值 |
|---|---|
| 总轮数 | 300 |
| 单卡batch size | 7 |
| 放大倍数 | 2 |
| 优化器 | Adam，betas=(0.5,0.999)，weight_decay=0 |
| 生成器初始学习率 | 5e-5 |
| 判别器初始学习率 | 1e-5 |
| 最低学习率 | 对应初始值的0.01倍 |
| 学习率调度 | 余弦调度，包含学习率预热 |
| 梯度范数上限 | 1 |
| 每轮保存间隔 | 1轮 |
| 数据加载进程 | 2 |

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 SRGAN_BATCH_SIZE=14 SRGAN_DATASET_DIR="./Unpaired henan" SRGAN_SAVE_DIR=logs_dual_experiment python train.py
```

DataParallel的batch是两卡总量，每卡7个样本。当前入口要求总batch可被可见GPU数量整除，且每卡至少2个样本；显存占用不保证恰好达到20GB。

## 5. 环境变量

| 名称 | 默认值 / 用途 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `0`，在导入PyTorch前设置，外部设置优先 |
| `SRGAN_EPOCHS` | `300` |
| `SRGAN_BATCH_SIZE` | `7` |
| `SRGAN_DATASET_DIR` | `./Unpaired henan`，三个列表从该目录读取；目录名含空格，在终端赋值时必须加引号 |
| `SRGAN_SAVE_DIR` | `logs`，新实验应使用独立目录 |
| `SRGAN_NUM_WORKERS` | `2` |
| `SRGAN_INIT_LR` | `5e-5` |
| `SRGAN_D_INIT_LR` | `1e-5` |
| `SRGAN_GRAD_CLIP` | `1.0` |
| `SRGAN_CYCLE_G1` | `1.0`，G1循环项的最终权重 |
| `SRGAN_CYCLE_G` | `0.1`，G循环项的最终权重 |
| `SRGAN_RESUME` | 默认空；新格式完整检查点路径 |
| `SRGAN_LOG_GRAPH` | `0`；设为1才尝试TensorBoard结构图记录 |

## 6. 权重、效果图与指标

每轮完成后，保存到 `SRGAN_SAVE_DIR`：

```text
实验目录/
├── G_EpochN-GLoss...-G1Loss....pth
├── G1_EpochN-GLoss...-G1Loss....pth
├── G_model_last_epoch_weights.pth
├── G1_model_last_epoch_weights.pth
├── training_last_checkpoint.pth
├── epoch_images/
│   ├── epoch_NNN.png
│   ├── epoch_NNN.json
│   └── epoch_NNN.npz
└── loss_日期_时间/
    ├── G_total_loss.txt、G1_total_loss.txt
    ├── G_image_loss.txt、G1_image_loss.txt
    ├── cycle_G_loss.txt、cycle_G1_loss.txt
    ├── D_total_loss.txt、D2_total_loss.txt
    ├── G_total_PSNR.txt、G_total_SSIM.txt
    ├── G1_total_PSNR.txt、G1_total_SSIM.txt
    ├── G_saturated_fraction.txt、G1_saturated_fraction.txt
    ├── sample_G1_eval_mse.txt、sample_G_eval_mse.txt
    ├── sample_G_eval_PSNR.txt、sample_G_eval_SSIM.txt
    ├── epoch_numbers.txt
    └── epoch_loss.png
```

- G/G1带轮数的文件分别保留；两个 `last_epoch` 推理权重每轮更新。
- `training_last_checkpoint.pth`每轮覆盖更新，包含六个模型、六个优化器、已完成轮数、损失阶段配置和学习率调度配置。
- PNG是固定训练样本0的五栏图：LR、NLR、G1去噪、G超分辨率和HR。
- JSON记录该样本评估模式下的数值范围、标准差、接近±1的比例和误差/指标。
- NPZ包含归一化的LR、NLR、去噪结果、SR和HR，不是原始物理振幅。
- TXT中的训练指标来自训练批次；`sample_*`来自固定训练样本的评估模式，二者不能等同于独立验证集成绩。

非有限输入、输出、损失或梯度会触发错误；固定样本近乎常量或饱和时会警告。这些保护不能保证训练不会退化。

## 7. 完整检查点续训

```bash
CUDA_VISIBLE_DEVICES=0 SRGAN_BATCH_SIZE=7 SRGAN_DATASET_DIR="./Unpaired henan" SRGAN_RESUME=logs_previous/training_last_checkpoint.pth SRGAN_SAVE_DIR=logs_continued python train.py
```

`SRGAN_RESUME`不能与源码中六个单独模型路径同时使用。加载后从已完成轮数的下一轮继续，并恢复六网络、优化器和原调度配置；其中保存的总轮数会恢复，不应假定更改 `SRGAN_EPOCHS` 就能延长旧调度。

没有显式设置的损失阶段环境变量会从检查点恢复，显式设置的变量优先。检查点没有保存所有随机数状态，不保证恢复后数据顺序或结果逐位复现。旧版只有G、G1权重时，无法完整恢复六网络训练。
