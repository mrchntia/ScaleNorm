from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNetwork(nn.Module):
    """
    Network used in the experiments on CIFAR-10
    Code adopted from: https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """

    def __init__(self, act_func=nn.Tanh, input_channels: int = 3):
        super(SmallNetwork, self).__init__()
        self.in_channels: int = input_channels

        # Variables to keep track of taken steps and samples in the model
        self.n_samples: int = 0
        self.n_steps: int = 0

        # Feature Layers
        feature_layer_config: List = [32,
                                      32, 'M',
                                      64,
                                      64, 'M',
                                      128,
                                      128, 'M']
        feature_layers: List = []

        c = self.in_channels
        for v in feature_layer_config:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'G':
                feature_layers += [nn.GroupNorm(num_groups=2, num_channels=c)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                feature_layers += [conv2d, act_func()]
                c = v
        self.features = nn.Sequential(*feature_layers)

        # Classifier Layers
        num_hidden: int = 128
        self.classifier = nn.Sequential(
            nn.Linear(c * 4 * 4, num_hidden), act_func(), nn.Linear(num_hidden, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def small_network(params):
    return SmallNetwork(params.act_func, params.input_channels)
