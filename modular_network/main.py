from typing import Union, List, Dict, Any, cast, Callable
import sys
import argparse
import random
import numpy as np
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from opacus import PrivacyEngine

from resnet_pytorch import resnet18, resnet34, resnet50
from densenet_pytorch import densenet121, densenet161, densenet169, densenet201
from vgg_pytorch import vgg11, vgg11_gn, vgg13, vgg13_gn, vgg16, vgg16_gn, vgg19, vgg19_gn


class Parameters:
    def __init__(self,
                 dataset: str,
                 sample_rate: float,
                 act_func: str,
                 model_arch: str,
                 epochs: int = 4,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 target_epsilon: float = 7.0,
                 grad_norm: float = 1.0,
                 sigma: float = 1.0
                 ):

        self.dataset: str = dataset
        self.epochs: int = epochs
        self.sample_rate: float = sample_rate
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.target_epsilon: float = target_epsilon
        self.grad_norm: float = grad_norm
        self.sigma: float = sigma
        self.model_arch: str = model_arch

        if act_func == 'tanh':
            self.act_func: Callable = nn.Tanh
        elif act_func == 'relu':
            self.act_func: Callable = nn.ReLU

        if self.dataset == 'CIFAR':
            self.input_channels: int = 3
            self.num_classes: int = 10

        # Fixed
        self.test_batch_size: int = 512
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.privacy: bool = True
        self.delta: float = 1e-5
        self.log_interval: int = 1
        self.secure_rng: bool = False
        self.group_norm: bool = None


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['CIFAR'])
    parser.add_argument('--act-func', type=str, default='tanh', choices=['tanh', 'relu'])
    parser.add_argument('--model-arch', type=str, default='densenet121', choices=[
        'vgg11',
        'vgg11_gn',
        'vgg13',
        'vgg13_gn',
        'vgg16',
        'vgg16_gn',
        'vgg19',
        'vgg19_gn',
        'resnet18',
        'resnet34',
        'resnet50',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201'
    ])
    parser.add_argument('--sample-rate', type=float, default=0.004)
    parser.add_argument('--test-batch-size', type=int, default=200)
    return parser.parse_args(args)


def get_dataloaders(params):
    if params.dataset == 'CIFAR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_loader = DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
            batch_size=int(params.sample_rate*50000)
        )
        test_loader = DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
            batch_size=params.test_batch_size,
            shuffle=False
        )

    return train_loader, test_loader


def train(model, train_loader, params, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(params.device), target.to(params.device)

        optimizer.zero_grad()

        # Forward and backward pass
        output = model(data)
        loss = criterion()(output, target)
        loss.backward()

        # Perform update
        optimizer.step()

        # Get current params form privacy engine
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(params.delta)
        grad_norm = optimizer.privacy_engine.max_grad_norm
        noise_multiplier = optimizer.privacy_engine.noise_multiplier

        # Print results
        if batch_idx % params.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            print(
                'ε: {:.4f}, grad norm: {:.4f}, sigma: {:.4f}'.format(
                    epsilon,
                    grad_norm,
                    noise_multiplier
                )
            )


def test(model, test_loader, device):
    model.eval()
    test_loss: float = 0
    correct: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output.shape())
            test_loss += torch.nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


nets: Dict[str, Callable] = {
    "vgg11": vgg11,
    "vgg11_gn": vgg11_gn,
    "vgg13": vgg13,
    "vgg13_gn": vgg13_gn,
    "vgg16": vgg16,
    "vgg16_gn": vgg16_gn,
    "vgg19": vgg19,
    "vgg19_gn": vgg19_gn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201
}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # set seed to make experiments deterministic
    seed: int = 50
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args(sys.argv[1:])
    params = Parameters(args.dataset, args.sample_rate, args.act_func, args.model_arch)

    model = nets[params.model_arch](params)
    print(model)

    train_loader, test_loader = get_dataloaders(params)

    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), params.learning_rate, params.momentum)

    # Init opacus privacy engine
    if params.privacy:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=params.sample_rate,
            alphas=[*range(2, 64)],
            epochs=params.epochs,
            target_delta=params.delta,
            target_epsilon=params.target_epsilon,
            noise_multiplier=params.sigma,
            max_grad_norm=params.grad_norm,
            secure_rng=params.secure_rng,
        )
        privacy_engine.attach(optimizer)
        privacy_engine.to(params.device)

    print("Training starts")
    for epoch in range(params.epochs):
        train(model, train_loader, params, optimizer, criterion)
        test(model, test_loader, params.device)
