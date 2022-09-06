import torch.nn as nn


def conv_block(
        in_channels, out_channels, pool=False, upsample=False, act_func=nn.Mish, norm_layer="batch", num_groups=None
):
    layers = []
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
    if norm_layer == "batch":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm_layer == "group":
        layers.append(nn.GroupNorm(min(num_groups, out_channels), out_channels))
    else:
        raise ValueError("`norm_layer` must be `batch` or `group`")
    layers.append(act_func())
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class LinkNet9(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 1,
            act_func: nn.Module = nn.Mish,
            scale_norm: bool = False,
            norm_layer: str = "batch",
            num_groups: int = 32,
    ):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups)
        self.conv2 = conv_block(64, 128, pool=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups)
        self.res1 = nn.Sequential(
            *[
                conv_block(128, 128, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups),
                conv_block(128, 128, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups),
            ]
        )
        self.conv3 = conv_block(128, 256, pool=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups)
        self.conv4 = conv_block(256, 256, pool=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups)
        self.res2 = nn.Sequential(
            *[
                conv_block(256, 256, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups),
                conv_block(256, 256, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups),
            ]
        )
        self.conv5 = conv_block(256, 512, pool=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups)
        self.deco1 = conv_block(
            512, 256, pool=False, upsample=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
        )
        self.res3 = nn.Sequential(
            *[
                conv_block(
                    256, 256, pool=False, upsample=False, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
                ),
                conv_block(
                    256, 256, pool=False, upsample=False, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
                ),
            ]
        )
        self.deco3 = conv_block(
            256, 256, pool=False, upsample=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
        )
        self.deco4 = conv_block(
            256, 128, pool=False, upsample=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
        )
        self.res4 = nn.Sequential(
            *[
                conv_block(
                    128, 128, pool=False, upsample=False, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
                ),
                conv_block(
                    128, 128, pool=False, upsample=False, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
                ),
            ]
        )
        self.deco6 = conv_block(
            128, 64, pool=False, upsample=True, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
        )
        self.head = conv_block(
            64, num_classes, pool=False, upsample=False, act_func=act_func, norm_layer=norm_layer, num_groups=num_groups
        )

        if scale_norm:
            self.scale_norm_1 = (
                nn.BatchNorm2d(128) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 128), 128)
            )
            self.scale_norm_2 = (
                nn.BatchNorm2d(256) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 256), 256)
            )
            self.scale_norm_3 = (
                nn.BatchNorm2d(256) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 256), 256)
            )
            self.scale_norm_4 = (
                nn.BatchNorm2d(256) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 256), 256)
            )
            self.scale_norm_5 = (
                nn.BatchNorm2d(128) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 128), 128)
            )
            self.scale_norm_6 = (
                nn.BatchNorm2d(128) if norm_layer == "batch" else nn.GroupNorm(min(num_groups, 128), 128)
            )

        else:
            self.scale_norm_1 = nn.Identity()
            self.scale_norm_2 = nn.Identity()
            self.scale_norm_3 = nn.Identity()
            self.scale_norm_4 = nn.Identity()
            self.scale_norm_5 = nn.Identity()
            self.scale_norm_6 = nn.Identity()

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        skip1 = out
        out = self.res1(out) + skip1
        out = self.scale_norm_1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        skip2 = out
        out = self.res2(out) + skip2
        out = self.scale_norm_2(out)
        out = self.conv5(out)

        out = self.deco1(out) + skip2
        out = self.scale_norm_3(out)
        out = self.res3(out) + out
        out = self.scale_norm_4(out)
        out = self.deco3(out)
        out = self.deco4(out) + skip1
        out = self.scale_norm_5(out)
        out = self.res4(out) + out
        out = self.scale_norm_6(out)
        out = self.deco6(out)
        out = self.head(out)
        return out
