from torch import cat
import torch.nn as nn
from src.diff_utils import ConvType


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map, conv_type):
        super(UNetDoubleConv, self).__init__()

        self.conv_type = conv_type

        self.conv1 = nn.Sequential(
            (
                nn.Conv2d(in_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']) if conv_type == ConvType.Conv2d else\
                nn.Conv3d(in_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation'])
            ),
            nn.LeakyReLU(0.2),
            (
                nn.BatchNorm2d(out_channels) if conv_type == ConvType.Conv2d else\
                nn.BatchNorm3d(out_channels)
            )
        )
        self.conv2 = nn.Sequential(
            (
                nn.Conv2d(out_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']) if conv_type == ConvType.Conv2d else\
                nn.Conv3d(out_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation'])
            ),
            nn.LeakyReLU(0.2),
            (
                nn.BatchNorm2d(out_channels) if conv_type == ConvType.Conv2d else\
                nn.BatchNorm3d(out_channels)
            )
        )
        self.time_map = nn.Sequential(
            nn.Linear(time_dimension, out_channels, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.cond_map = nn.Sequential(
            nn.Linear(cond_dimension, out_channels, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False) if conv_type == ConvType.Conv2d else\
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        )
        self.act = nn.LeakyReLU(0.2)
        self.batch_norm = (
            nn.BatchNorm2d(out_channels) if conv_type == ConvType.Conv2d else\
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x, time_embedding, y):
        if self.conv_type == ConvType.Conv2d:
            dc_new = self.conv2(self.conv1(x) + self.cond_map(y)[:,:,None,None] + self.time_map(time_embedding)[:,:,None,None])
        else:
            dc_new = self.conv2(self.conv1(x) + self.cond_map(y)[:,:,None,None,None] + self.time_map(time_embedding)[:,:,None,None,None])
        return self.batch_norm(self.act(dc_new + self.res_conv(x)))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map, conv_type):
        super(UNetDownBlock, self).__init__()

        self.double_conv = UNetDoubleConv(in_channels, out_channels, time_dimension, cond_dimension, conv_map, conv_type)
        self.pool = (
            nn.Conv2d(out_channels, out_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride']) if conv_type == ConvType.Conv2d else\
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        )
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x, time_embedding, y):
        conved = self.double_conv(x, time_embedding, y)
        return conved, self.pool(conved)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map, conv_type):
        super(UNetUpBlock, self).__init__()

        self.unpool = (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride']) if conv_type == ConvType.Conv2d else\
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        )
        self.double_conv = UNetDoubleConv(2 * in_channels, out_channels, time_dimension, cond_dimension, conv_map, conv_type)

        self.attention_conv = (
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0) if conv_type == ConvType.Conv2d else\
            nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = (
            nn.BatchNorm2d(in_channels) if conv_type == ConvType.Conv2d else\
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x, x_prior, time_embedding, y):
        attn_x_prior = self.batch_norm(x_prior * self.sigmoid(self.attention_conv(x_prior)))
        return self.double_conv(cat((attn_x_prior, self.unpool(x)), dim=1), time_embedding, y)


class UNet(nn.Module):
    def __init__(self, in_channels, time_dimension, cond_dimension, conv_map, conv_type):
        super(UNet, self).__init__()

        self.down1 = UNetDownBlock(in_channels, 64, time_dimension, cond_dimension, conv_map, conv_type)
        self.down2 = UNetDownBlock(64, 128, time_dimension, cond_dimension, conv_map, conv_type)
        self.down3 = UNetDownBlock(128, 256, time_dimension, cond_dimension, conv_map, conv_type)
        self.down4 = UNetDownBlock(256, 512, time_dimension, cond_dimension, conv_map, conv_type)
        self.layer5 = UNetDoubleConv(512, 512, time_dimension, cond_dimension, conv_map, conv_type)
        self.up4 = UNetUpBlock(512, 256, time_dimension, cond_dimension, conv_map, conv_type)
        self.up3 = UNetUpBlock(256, 128, time_dimension, cond_dimension, conv_map, conv_type)
        self.up2 = UNetUpBlock(128, 64, time_dimension, cond_dimension, conv_map, conv_type)
        self.up1 = UNetUpBlock(64, in_channels, time_dimension, cond_dimension, conv_map, conv_type)
        self.last = (
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True) if conv_type == ConvType.Conv2d else\
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
            
    def forward(self, x, time_embedding, y):
        x1, x_run = self.down1(x, time_embedding, y)
        x2, x_run = self.down2(x_run, time_embedding, y)
        x3, x_run = self.down3(x_run, time_embedding, y)
        x4, x_run = self.down4(x_run, time_embedding, y)
        x_run = self.layer5(x_run, time_embedding, y)
        x_run = self.up4(x_run, x4, time_embedding, y)
        x_run = self.up3(x_run, x3, time_embedding, y)
        x_run = self.up2(x_run, x2, time_embedding, y)
        return self.last(self.up1(x_run, x1, time_embedding, y))