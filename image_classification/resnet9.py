from torch import nn
import warnings


def conv_bn_act(
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        act_func: nn.Module = nn.Mish,
        num_groups: int = None
):
    """Conv-BatchNorm-Activation
    """
    if num_groups is not None:
        warnings.warn("num_groups has no effect with BatchNorm")
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_gn_act(
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        act_func: nn.Module = nn.Mish,
        num_groups: int = 32
):
    """Conv-GroupNorm-Activation
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(num_groups, out_channels), out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        act_func: nn.Module = nn.Mish,
        scale_norm: bool = False,
        norm_layer: str = "batch",
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC

        Args:
            in_channels (int, optional): Channels in the input image. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
            act_func (nn.Module, optional): Activation function to use. Defaults to nn.Mish.
            scale_norm (bool, optional): Whether to add an extra normalisation layer after each residual block. \
            Defaults to False.
            norm_layer (str, optional): Normalisation layer. One of `batch` or `group`. Defaults to "batch".
            num_groups (tuple[int], optional): Number of groups in GroupNorm layers.\
            Must be a tuple with 4 elements, corresponding to the GN layer in the first conv block, \
            the first res block, the second conv block and the second res block. Defaults to (32, 32, 32, 32).
        """
        super().__init__()

        if norm_layer == "batch":
            conv_block = conv_bn_act
        elif norm_layer == "group":
            conv_block = conv_gn_act
        else:
            raise ValueError("`norm_layer` must be `batch` or `group`")

        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_block(
            in_channels, 64, act_func=act_func, num_groups=groups[0]
        )
        self.conv2 = conv_block(
            64, 128, pool=True, act_func=act_func, num_groups=groups[0]
        )

        self.res1 = nn.Sequential(
            *[
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
            ]
        )

        self.conv3 = conv_block(
            128, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )
        self.conv4 = conv_block(
            256, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )

        self.res2 = nn.Sequential(
            *[
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
            ]
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                nn.BatchNorm2d(128)
                if norm_layer == "batch"
                else nn.GroupNorm(min(num_groups[1], 128), 128)
            )  # type:ignore
            self.scale_norm_2 = (
                nn.BatchNorm2d(256)
                if norm_layer == "batch"
                else nn.GroupNorm(min(groups[3], 256), 256)
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


class ResNet9LRF(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        act_func: nn.Module = nn.Mish,
        scale_norm: bool = False,
        norm_layer: str = "batch",
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC

        This variant has more aggressive downscaling for use with larger images.

        Args:
            in_channels (int, optional): Channels in the input image. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
            act_func (nn.Module, optional): Activation function to use. Defaults to nn.Mish.
            scale_norm (bool, optional): Whether to add an extra normalisation layer after each residual block. Defaults to False.
            norm_layer (str, optional): Normalisation layer. One of `batch` or `group`. Defaults to "batch".
            num_groups (tuple[int], optional): Number of groups in GroupNorm layers.\
            Must be a tuple with 4 elements, corresponding to the GN layer in the first conv block, \
            the first res block, the second conv block and the second res block. Defaults to (32, 32, 32, 32).
        """
        super().__init__()

        if norm_layer == "batch":
            conv_block = conv_bn_act
        elif norm_layer == "group":
            conv_block = conv_gn_act
        else:
            raise ValueError("`norm_layer` must be `batch` or `group`")

        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_block(
            in_channels, 64, pool=True, act_func=act_func, num_groups=groups[0]
        )
        self.conv2 = nn.Sequential(
            *[
                conv_block(64, 128, pool=True, act_func=act_func, num_groups=groups[0]),
                conv_block(
                    128, 128, pool=True, act_func=act_func, num_groups=groups[0]
                ),
            ]
        )

        self.res1 = nn.Sequential(
            *[
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
            ]
        )

        self.conv3 = conv_block(
            128, 128, pool=True, act_func=act_func, num_groups=groups[2]
        )
        self.conv4 = nn.Sequential(
            *[
                conv_block(
                    128, 256, pool=True, act_func=act_func, num_groups=groups[2]
                ),
                conv_block(
                    256, 256, pool=True, act_func=act_func, num_groups=groups[2]
                ),
            ]
        )

        self.res2 = nn.Sequential(
            *[
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
            ]
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                nn.BatchNorm2d(128)
                if norm_layer == "batch"
                else nn.GroupNorm(min(groups[1], 128), 128)
            )  # type:ignore
            self.scale_norm_2 = (
                nn.BatchNorm2d(256)
                if norm_layer == "batch"
                else nn.GroupNorm(min(groups[3], 256), 256)
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
