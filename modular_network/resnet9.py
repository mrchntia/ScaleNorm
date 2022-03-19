from torch import nn
from typing import Union
import warnings


def conv_bn_act(
    in_channels, out_channels, pool=False, pool_no=2, act_func=nn.Mish, num_groups=None
):
    if num_groups is not None:
        warnings.warn("num_groups has no effect with BatchNorm")
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


def conv_gn_act(
    in_channels, out_channels, pool=False, pool_no=2, act_func=nn.Mish, num_groups=32
):
    """Conv-GroupNorm-Activation
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        act_func=nn.Mish,
        scale_norm=False,
        norm_layer="batch",
        num_groups: Union[int, list[int]] = 32,
    ):
        super().__init__()

        if norm_layer == "batch":
            conv_block = conv_bn_act
        elif norm_layer == "group":
            conv_block = conv_gn_act
        else:
            raise ValueError("`norm_layer` must be `batch` or `group`")

        if isinstance(num_groups, int):
            if scale_norm:
                groups = [num_groups for _ in range(10)]
            else:
                groups = [num_groups for _ in range(8)]
        elif isinstance(num_groups, list):
            if scale_norm:
                assert (
                    len(num_groups) == 10
                ), "`num_groups` must have exactly 10 members with scale_norm (8 for the layers, 2 for the residual norms)"
            else:

                assert (
                    len(num_groups) == 8
                ), "`num_groups` must have exactly 8 members without scale_norm"
            groups = num_groups
        else:
            raise TypeError("`num_groups` must be an integer or list.")

        self.conv1 = conv_block(
            in_channels, 64, act_func=act_func, num_groups=groups[0]
        )
        self.conv2 = conv_block(
            64, 128, pool=True, pool_no=2, act_func=act_func, num_groups=groups[1]
        )
        self.res1 = nn.Sequential(
            conv_block(128, 128, act_func=act_func, num_groups=groups[2]),
            conv_block(128, 128, act_func=act_func, num_groups=groups[3]),
        )

        self.conv3 = conv_block(
            128, 256, pool=True, act_func=act_func, num_groups=groups[4]
        )
        self.conv4 = conv_block(
            256, 256, pool=True, pool_no=2, act_func=act_func, num_groups=groups[5]
        )
        self.res2 = nn.Sequential(
            conv_block(256, 256, act_func=act_func, num_groups=groups[6]),
            conv_block(256, 256, act_func=act_func, num_groups=groups[7]),
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                nn.BatchNorm2d(128)
                if norm_layer == "batch"
                else nn.GroupNorm(groups[8], 128)
            )  # type:ignore
            self.scale_norm_2 = (
                nn.BatchNorm2d(256)
                if norm_layer == "batch"
                else nn.GroupNorm(groups[9], 256)
            )  # type:ignore
        else:
            self.scale_norm_1 = nn.Identity()  # type:ignore
            self.scale_norm_2 = nn.Identity()  # type:ignore

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.scale_norm_1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.scale_norm_2(out)
        out = self.MP(out)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out


def resnet9(params) -> ResNet9:
    return ResNet9(
        params.in_channels,
        params.num_classes,
        params.act_func,
        params.scale_norm,
        params.norm_layer
    )

