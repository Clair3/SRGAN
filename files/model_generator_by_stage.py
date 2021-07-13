import math
import torch
from torch import nn
import numpy as np
import Potts as pt


class Generator_stage(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2)) # 2 pour factor 2, 3 pour factor 8
        n_residual_blocks = 8
        super(Generator_stage, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU())

        self.block1_bis = nn.Sequential(
            nn.PReLU())
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample = nn.Sequential(UpsampleBLock(64, 2))

        self.block8 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=9, padding=4))

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample = nn.Sequential(UpsampleBLock(64, 2))

        self.block1_2 = nn.Sequential(
            nn.PReLU())

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks_2 = nn.Sequential(*res_blocks)
        self.block7_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample_2 = nn.Sequential(UpsampleBLock(64, 2))

        self.block1_3 = nn.Sequential(
            nn.PReLU())

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks_3 = nn.Sequential(*res_blocks)
        self.block7_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample_3 = nn.Sequential(UpsampleBLock(64, 2))

        self.block8 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=9, padding=4))

    def forward(self, x):
        block1 = self.block1(x)
        block_res = self.res_blocks(block1)
        block7 = self.block7(block_res)
        block_upsample = self.block_upsample(block1 + block7)

        block1 = self.block1_2(block_upsample)
        block_res = self.res_blocks_2(block1)
        block7 = self.block7_2(block_res)
        block_upsample = self.block_upsample_2(block1 + block7)

        block1 = self.block1_3(block_upsample)
        block_res = self.res_blocks_3(block1)
        block7 = self.block7_3(block_res)
        block_upsample = self.block_upsample_3(block1 + block7)

        block8 = self.block8(block_upsample)
        out = (torch.tanh(block8) + 1) / 2
        return out

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))  # 2 pour factor 2, 3 pour factor 8
        n_residual_blocks = 8
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU())
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample = nn.Sequential(UpsampleBLock(64, 2))
        self.block8 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=9, padding=4))

        self.block1_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU())
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks_2 = nn.Sequential(*res_blocks)
        self.block7_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample_2 = nn.Sequential(UpsampleBLock(64, 2))
        self.block8_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=9, padding=4))

        self.block1_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU())
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks_3 = nn.Sequential(*res_blocks)
        self.block7_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        self.block_upsample_3 = nn.Sequential(UpsampleBLock(64, 2))
        self.block8_3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=9, padding=4))

    def forward(self, x):
        block1 = self.block1(x)
        block_res = self.res_blocks(block1)
        block7 = self.block7(block_res)
        block_upsample = self.block_upsample(block1 + block7)
        block8 = self.block8(block_upsample)

        block1 = self.block1_2(block8)
        block_res = self.res_blocks_2(block1)
        block7 = self.block7_2(block_res)
        block_upsample = self.block_upsample_2(block1 + block7)
        block8 = self.block8_2(block_upsample)

        block1 = self.block1_3(block8)
        block_res = self.res_blocks_3(block1)
        block7 = self.block7_3(block_res)
        block_upsample = self.block_upsample_3(block1 + block7)
        block8 = self.block8_3(block_upsample)

        out = (torch.tanh(block8) + 1) / 2
        return out




class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels

        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        reduction = 16
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.se = SELayer(channels, reduction)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        # residual = self.se(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


