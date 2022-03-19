import torch
from torch import nn
import warnings
from typing import Optional


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0,
                 dilation: int = 1, norm: str = "group", groups: Optional[int] = 32, activation: nn.Module = nn.Mish):
        super().__init__()
        assert norm in {"group", "batch"}, "norm must be `group` or `batch`"

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        if norm == "group":
            self.norm = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels, affine=True)
        else:
            if groups is not None:
                warnings.warn("`groups` has no effect when `norm` is `batch`")
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# MoNet uses transposed convolutions but opacus doesn't support these, so we're
# using upsampling
class Upsample(nn.Module):
    def __init__(self, mode: str = "nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)

    def forward(self, x):
        return self.upsample(x)


class RepeatBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: str = "group",
            groups: Optional[int] = 32,
            activation: nn.Module = nn.Mish,
            dropout_rate: float = 0.2,
            scale_norm: bool = True,
    ):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, dilation=4, padding=4, norm=norm, groups=groups,
                                 activation=activation)
        self.conv2 = ConvNormAct(out_channels, out_channels, dilation=3, padding=3, norm=norm, groups=groups,
                                 activation=activation)
        self.conv2 = ConvNormAct(out_channels, out_channels, dilation=2, padding=2, norm=norm, groups=groups,
                                 activation=activation)
        self.conv3 = ConvNormAct(out_channels, out_channels, dilation=1, padding=1, norm=norm, groups=groups,
                                 activation=activation)
        self.dropout = nn.Dropout2d(p=dropout_rate)  # equivalent to spatial dropout in TF

        if scale_norm:
            self.scale_norm = nn.GroupNorm(
                num_groups=min(groups, out_channels),
                num_channels=out_channels
            ) if norm == "group" else nn.BatchNorm2d(out_channels)
        else:
            self.scale_norm = nn.Identity()

    def forward(self, x):
        skip = x
        c1 = self.conv1(x)
        c1 = self.dropout(c1)
        s1 = self.scale_norm(c1+skip)
        c2 = self.conv2(s1)
        c2 = self.dropout(c2)
        c3 = self.conv3(c2)
        c3 = self.dropout(c3)
        s2 = self.scale_norm(c2+c3)
        c4 = self.conv3(s2)
        return c4


class Pooling(nn.Module):
    def __init__(self, mode="max"):
        super().__init__()
        assert mode in ("max", "average"), "mode must be max or average"
        if mode == "max":
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)


class MoNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, initial_filters: int = 16, norm: str = "group",
                 groups: int = 32, activation: nn.Module = nn.Mish, dropout_enc: float = 0.2, dropout_dec: float = 0.2,
                 upsample_mode: str = "nearest", scale_norm: bool = True, pool_mode: str = "max"):
        super().__init__()
        assert (initial_filters % 2 == 0), "`initial filters` must be divisible by 2"

        layout = {"input": in_channels, "enc_1": initial_filters * 2, "enc_2": initial_filters * 2,
                  "bneck": initial_filters * 4, "dec_1": initial_filters * 2, "dec_2": initial_filters,
                  "out": out_channels}

        self.encoder_block_1 = nn.Sequential(*[
            ConvNormAct(in_channels=layout["input"], out_channels=layout["enc_1"], padding=1, norm=norm, groups=groups,
                        activation=activation),
            RepeatBlock(in_channels=layout["enc_1"], out_channels=layout["enc_1"], norm=norm, groups=groups,
                        activation=activation, dropout_rate=dropout_enc, scale_norm=scale_norm)])

        self.pool_1 = Pooling(mode=pool_mode)

        self.encoder_block_2 = nn.Sequential(*[
            ConvNormAct(in_channels=layout["enc_1"], out_channels=layout["enc_2"], padding=1, norm=norm, groups=groups,
                        activation=activation),
            RepeatBlock(in_channels=layout["enc_2"], out_channels=layout["enc_2"], norm=norm, groups=groups,
                        activation=activation, dropout_rate=dropout_enc, scale_norm=scale_norm)])

        self.pool_2 = Pooling(mode=pool_mode)

        self.bottleneck = nn.Sequential(*[
            ConvNormAct(in_channels=layout["enc_2"], out_channels=layout["bneck"], padding=1, norm=norm, groups=groups,
                        activation=activation),
            RepeatBlock(in_channels=layout["bneck"], out_channels=layout["bneck"], norm=norm, groups=groups,
                        activation=activation, dropout_rate=dropout_enc, scale_norm=scale_norm)])

        self.unpool_1 = Upsample(mode=upsample_mode)

        self.decoder_block_1 = nn.Sequential(*[
            ConvNormAct(in_channels=layout["bneck"] + layout["enc_2"], out_channels=layout["dec_1"], padding=1,
                        norm=norm, groups=groups, activation=activation),
            RepeatBlock(in_channels=layout["dec_1"], out_channels=layout["dec_1"], norm=norm, groups=groups,
                        activation=activation, dropout_rate=dropout_dec, scale_norm=scale_norm)])

        self.unpool_2 = Upsample(mode=upsample_mode)

        self.decoder_block_2 = nn.Sequential(*[
            ConvNormAct(in_channels=layout["dec_1"] + layout["enc_1"], out_channels=layout["dec_2"], padding=1,
                        norm=norm, groups=groups, activation=activation),
            RepeatBlock(in_channels=layout["dec_2"], out_channels=layout["dec_2"], norm=norm, groups=groups,
                        activation=activation, dropout_rate=dropout_dec, scale_norm=scale_norm)])

        self.head = nn.Sequential(*[
            nn.Conv2d(in_channels=layout["dec_2"], out_channels=layout["out"], kernel_size=3, stride=1, padding=1,
                      dilation=1, bias=False),
            activation()])
        if scale_norm:
            self.scale_norm_1 = nn.GroupNorm(num_groups=groups, num_channels=layout["bneck"] + layout[
                "enc_2"]) if norm == "group" else nn.BatchNorm2d(layout["bneck"] + layout["enc_2"])
            self.scale_norm_2 = nn.GroupNorm(num_groups=groups, num_channels=layout["dec_1"] + layout[
                "enc_1"]) if norm == "group" else nn.BatchNorm2d(layout["dec_1"] + layout["enc_1"])
        else:
            self.scale_norm_1 = nn.Identity()
            self.scale_norm_2 = nn.Identity()

    def forward(self, x):
        x = self.encoder_block_1(x)
        skip1 = x.clone()
        x = self.pool_1(x)
        x = self.encoder_block_2(x)
        skip2 = x.clone()
        x = self.pool_2(x)
        x = self.bottleneck(x)
        x = self.unpool_1(x)
        x = self.scale_norm_1(torch.cat([x, skip2], dim=1))
        x = self.decoder_block_1(x)
        x = self.unpool_2(x)
        x = self.scale_norm_2(torch.cat([x, skip1], dim=1))
        x = self.decoder_block_2(x)
        return self.head(x)
