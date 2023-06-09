import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True,
                              track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True,
                              track_running_stats=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_features=64, num_residuals=6, c_dim=5):
        super(Generator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3+c_dim, num_features, kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(num_features, affine=True,
                              track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        for i in range(2):
            self.down_blocks.append(nn.Conv2d(
                num_features, num_features*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.down_blocks.append(nn.InstanceNorm2d(
                num_features*2, affine=True, track_running_stats=True))
            self.down_blocks.append(nn.ReLU(inplace=True))
            num_features = num_features * 2

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList()
        for i in range(2):
            self.up_blocks.append(nn.ConvTranspose2d(
                num_features, num_features//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_blocks.append(nn.InstanceNorm2d(
                num_features//2, affine=True, track_running_stats=True))
            self.up_blocks.append(nn.ReLU(inplace=True))
            num_features = num_features // 2

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features, 3, kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.initial_conv(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.res_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.final_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, img_size=128, num_feature=64, num_repeat=6, c_dim=5):
        super(Discriminator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        )

        self.model = nn.ModuleList()

        for i in range(1, num_repeat):
            self.model.append(
                nn.Conv2d(num_feature, num_feature*2, kernel_size=4, stride=2, padding=1))
            self.model.append(nn.LeakyReLU(0.01))
            num_feature = num_feature * 2

        kernel_size = int(img_size / np.power(2, num_repeat))
        self.final_conv1 = nn.Conv2d(
            num_feature, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_conv2 = nn.Conv2d(
            num_feature, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        x = self.initial_conv(x)

        for layer in self.model:
            x = layer(x)

        out_src = self.final_conv1(x)
        out_cls = self.final_conv2(x)

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
