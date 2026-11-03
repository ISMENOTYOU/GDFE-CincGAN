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

当前入口默认 `Cuda=True`，需要可用CUDA环境。
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

DataParallel的batch是两卡总量，每卡7个样本。当前入口要求总batch可被可见GPU数量整除，且每卡至少2个样本；

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
- TXT中的训练指标来自训练批次；`sample_*`来自固定训练样本的评估模式，二者不能等同于独立验证集成绩。

## 7. 完整检查点续训

```bash
CUDA_VISIBLE_DEVICES=0 SRGAN_BATCH_SIZE=7 SRGAN_DATASET_DIR="./Unpaired henan" SRGAN_RESUME=logs_previous/training_last_checkpoint.pth SRGAN_SAVE_DIR=logs_continued python train.py
```
`SRGAN_RESUME`不能与源码中六个单独模型路径同时使用。加载后从已完成轮数的下一轮继续，并恢复六网络、优化器和原调度配置；
