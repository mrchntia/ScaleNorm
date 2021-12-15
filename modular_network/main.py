from typing import Union, List, Dict, Any, cast, Callable
import sys
import argparse
import random
import numpy as np
import warnings
import uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from opacus import PrivacyEngine

from resnet_pytorch import resnet18, resnet34, resnet50
from resnet_sn import resnet18_sn
from densenet_pytorch import densenet121, densenet161, densenet169, densenet201
from vgg_pytorch import vgg11, vgg11_gn, vgg13, vgg13_gn, vgg16, vgg16_gn, vgg19, vgg19_gn
from small_conv_net import small_network

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
    "resnet18_sn": resnet18_sn,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "smallnet": small_network
}

act_funcs: Dict[str, Callable] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU
}


class Parameters:
    def __init__(self,
                 dataset: str,
                 sample_rate: float,
                 act_func: str,
                 model_arch: str,
                 epochs: int,
                 learning_rate: float,
                 momentum: float,
                 target_epsilon: float,
                 grad_norm: float,
                 noise_mult: float,
                 privacy: bool
                 ):
        self.dataset: str = dataset
        self.epochs: int = epochs
        self.sample_rate: float = sample_rate
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.target_epsilon: float = target_epsilon
        self.grad_norm: float = grad_norm
        self.noise_mult: float = noise_mult
        self.model_arch: str = model_arch
        self.privacy: bool = privacy
        self.act_func: Callable = act_funcs[act_func]

        if self.dataset == 'CIFAR':
            self.input_channels: int = 3
            self.num_classes: int = 10

        # Fixed
        self.test_batch_size: int = 512
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_delta: float = 1e-5
        self.log_interval: int = 20
        self.secure_rng: bool = False
        self.group_norm: bool = False


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['CIFAR'])
    parser.add_argument('--model-arch', type=str, default='vgg11', choices=[
        'vgg11',
        'vgg11_gn',
        'vgg13',
        'vgg13_gn',
        'vgg16',
        'vgg16_gn',
        'vgg19',
        'vgg19_gn',
        'resnet18',
        'resnet18_sn',
        'resnet34',
        'resnet50',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201',
        'smallnet'
    ])
    parser.add_argument('--sample-rate', type=float, default=0.004)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--act-func', type=str, default='tanh', choices=['tanh', 'relu', 'selu'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--target-epsilon', type=float, default=None)
    parser.add_argument('--noise-mult', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad-norm', type=float, default=1.0)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
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
            batch_size=int(params.sample_rate * 50000)
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
        output = model(data)
        loss = criterion()(output, target)
        loss.backward()
        optimizer.step()

        if params.privacy:
            epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=params.target_delta)
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
            if params.privacy:
                print(
                    'ε: {:.4f}, grad norm: {:.4f}'.format(
                        epsilon,
                        params.grad_norm,
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
    return 100.0 * correct / len(test_loader.dataset)


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
    params = Parameters(
        args.dataset,
        args.sample_rate,
        args.act_func,
        args.model_arch,
        args.epochs,
        args.learning_rate,
        args.momentum,
        args.target_epsilon,
        args.grad_norm,
        args.noise_mult,
        args.privacy
    )

    model: nn.Module = nets[params.model_arch](params).to(params.device)
    print(model)

    train_loader, test_loader = get_dataloaders(params)

    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), params.learning_rate, params.momentum)

    if params.privacy:
        privacy_engine = PrivacyEngine()
        if params.target_epsilon:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=params.target_epsilon,
                target_delta=params.target_delta,
                epochs=params.epochs,
                max_grad_norm=params.grad_norm
            )

        else:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=params.noise_mult,
                max_grad_norm=params.grad_norm,
            )

    print("Training starts")
    acc_list: List = []
    for epoch in range(params.epochs):
        train(model, train_loader, params, optimizer, criterion)
        acc = test(model, test_loader, params.device)
        acc_list.append(acc)

    if params.privacy:
        final_epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=params.target_delta)
    else:
        final_epsilon, best_alpha = None, None

    file_name = "{}_{}_{:.2f}_{}_{}_model".format(
        params.model_arch,
        params.epochs,
        final_epsilon,
        params.learning_rate,
        params.grad_norm
    )
    # torch.save(model, "exp03/{}".format(file_name))
    file = open("exp03/summary_exp03.txt", "a")
    file.write("Model:  {}\n".format(file_name))
    file.write(
        'dataset: {}, model_arch: {}, act_func: {}, \nlearning_rate: {}, sample_rate: {}, epochs: {}, momentum: {}, \n'.format(
            params.dataset,
            params.model_arch,
            args.act_func,
            params.learning_rate,
            params.sample_rate,
            params.epochs,
            params.momentum,
        )
    )
    if params.privacy:
        file.write(
            'target_epsilon: {}, grad_norm: {}, noise_mult: {}\nfinal_epsilon: {:.2f}, final_grad_norm: {:.2f}, \n'.format(
                params.target_epsilon,
                params.grad_norm,
                params.noise_mult,
                final_epsilon,
                params.grad_norm,
            )
        )
    file.write('test_acc: {}\n\n'.format(acc_list))
    file.close()
