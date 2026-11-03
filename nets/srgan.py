import math
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='low-training-memory'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        expand, linear = 6, 0.8

        self.body1 = nn.Sequential(
            nn.Conv2d(channels, channels * expand, 1),
            nn.PReLU(channels * expand),
            nn.Conv2d(channels * expand, int(channels * linear), 1),
            nn.Conv2d(int(channels * linear), channels, 1)
        )
        self.body2 = nn.Sequential(
            ShiftConv2d(channels, channels),
            nn.PReLU(channels),
            ShiftConv2d(channels, channels)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(channels)
        self.bn = nn.BatchNorm2d(channels)
        # self.dense = nn.Linear(channels, channels // 3)
        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        skip = x
        x1 = self.body1(x)
        x2 = self.body2(x)
        x3 = x
        x4 = x1 + x2 + x3
        x4 = self.pool(x4)
        x4 = self.conv(x4)
        x4 = self.bn(x4)
        x4 = self.prelu(x4)
        x4 = self.conv(x4)
        # x4 = x4.view(x4.size(0), -1)
        # x41 = self.dense(x4)
        # x42 = self.dense(x4)
        # x43 = self.dense(x4)
        x4 = self.softmax(x4)
        # x42 = self.softmax(x4)
        # x43 = self.softmax(x4)
        x41 = x1 * x4
        x42 = x2 * x4
        x5 = x41 + x42

        return x5 + skip


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 添加第一个空洞卷积层
        self.dilated_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 添加第二个空洞卷积层
        self.dilated_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        # x = self.dilated_conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.conv2(x)
        # x = self.dilated_conv2(x)
        x = self.bn2(x)

        return x + short_cut


class DPB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPB, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

        self.gelu = nn.GELU()
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        skip = x
        x = self.depth_conv(x)
        x = self.point_conv(x)

        # Skip Connection
        x = skip + x  # 再加上dw_out

        # GELU activation
        out = self.gelu(x)

        return out


class FEB(nn.Module):
    def __init__(self, channels):
        super(FEB, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.depth_conv = nn.Conv2d(channels,
                                    channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=channels)
        self.point_conv = nn.Conv2d(channels,
                                    channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

        self.gelu = nn.GELU()

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.sigmoid(conv_out)  # Sigmoid 激活
        conv_out_2 = self.depth_conv(x)
        conv_out_2 = self.point_conv(conv_out_2)
        conv_out_2 = self.gelu(conv_out_2)
        output = conv_out * conv_out_2  # 两条路径相乘
        return output + x  # 加上输入


class FE(nn.Module):
    def __init__(self, channels):
        super(FE, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.feb = FEB(channels)

    def forward(self, x):
        x = self.conv(x)
        out1 = x
        x = self.feb(x)
        out2 = x
        x = self.feb(x)
        out3 = x
        x = self.feb(x)
        out4 = x
        out = out1 + out2 + out3 + out4

        return out


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            conv_weight_rd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class DEAB(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DEAB, self).__init__()

        self.conv1 = DEConv(dim)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Generatordown(nn.Module):
    def __init__(self, num_residual=16):
        super(Generatordown, self).__init__()

        self.block_in = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(64),
        )
        self.blocks = []
        for _ in range(num_residual):
            self.blocks.append(ResidualBlock(64))
        self.blocks = nn.Sequential(*self.blocks)
        self.block_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.downsample = DownsampleBlock(64)
        self.out = nn.Conv2d(64, 1, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.block_in(x)
        short_cut = x
        x = self.blocks(x)
        x = self.block_out(x)
        x = self.downsample(x + short_cut)
        x = self.out(x)
        return x


class Generatorup(nn.Module):
    def __init__(self, scale_factor, num_residual=16):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generatorup, self).__init__()

        self.block_in = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(64),
        )
        self.blocks = []
        for _ in range(num_residual):
            self.blocks.append(ResidualBlock(64))
        self.blocks = nn.Sequential(*self.blocks)
        self.block_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upsample.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.upsample = nn.Sequential(*self.upsample)
        self.deab = DEAB(64)
        self.fe = FE(64)

    def forward(self, x):
        x = self.block_in(x)
        short_cut = x
        x = self.blocks(x)
        x = self.deab(x)
        x = self.fe(x)
        x = self.block_out(x)
        upsample = self.upsample(x + short_cut)
        return torch.tanh(upsample)


class Generator(nn.Module):
    def __init__(self, num_residual=12):
        super(Generator, self).__init__()

        self.block_in = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(64),
        )
        self.blocks = []
        for _ in range(num_residual):
            self.blocks.append(ResidualBlock(64))
        self.blocks = nn.Sequential(*self.blocks)
        self.block_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 1, kernel_size=9, padding=4)
        )

    def forward(self, x):
        x = self.block_in(x)
        short_cut = x
        x = self.blocks(x)
        x = self.block_out(x)
        x = self.Conv(x + short_cut)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),  # 512, 1024, kernel_size=1
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)
