from typing import Union, List, Dict, Any, cast, Callable

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, init_weights: bool = True) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], group_norm: bool = False, act_func: Callable = nn.Tanh) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if group_norm:
                layers += [conv2d, nn.GroupNorm(min(32, v), v), act_func()]
            else:
                layers += [conv2d, act_func()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["11"],  group_norm=False, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg11_gn(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["11"], group_norm=True, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg13(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["13"], group_norm=False, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg13_gn(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["13"], group_norm=True, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg16(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["16"], group_norm=False, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg16_gn(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["16"], group_norm=True, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg19(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["19"], group_norm=False, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )


def vgg19_gn(params, **kwargs: Any) -> VGG:
    return VGG(
        make_layers(cfgs["19"], group_norm=True, act_func=params.act_func),
        num_classes=params.num_classes,
        **kwargs
    )
