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

from resnet_pytorch import resnet18
from densenet_pytorch import densenet121
from vgg_pytorch import vgg11, vgg11_gn


class Parameters:
    def __init__(self,
                 dataset,
                 sample_rate,
                 act_func,
                 model_arch,
                 epochs=4,
                 learning_rate=0.1,
                 momentum=0.9,
                 target_epsilon=7.0,
                 grad_norm=1.0,
                 sigma=1.0
                 ):

        self.dataset = dataset
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.target_epsilon = target_epsilon
        self.grad_norm = grad_norm
        self.sigma = sigma
        self.model_arch = model_arch

        if act_func == 'tanh':
            self.act_func = nn.Tanh
        elif act_func == 'relu':
            self.act_func = nn.ReLU

        if self.dataset == 'CIFAR':
            self.input_channels = 3
            self.num_classes = 10

        # Fixed
        self.test_batch_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.privacy = True
        self.delta = 1e-5
        self.log_interval = 1
        self.input_channels = 3
        self.secure_rng = False
        self.group_norm = None


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['CIFAR'])
    parser.add_argument('--act-func', type=str, default='tanh', choices=['tanh', 'relu'])
    parser.add_argument('--model-arch', type=str, default='densenet121', choices=[
        'vgg11',
        'vgg11_gn',
        'resnet18',
        'densenet121'
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
    test_loss = 0
    correct = 0
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
    "vgg11_bn": vgg11_gn,
    "resnet18": resnet18,
    "densenet121": densenet121
}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # set seed to make experiments deterministic
    seed = 50
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
