import argparse
import random
import warnings

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from utils import get_tiny_dataloader, get_cifar_dataloader, save_checkpoint, load_checkpoint, train_one_epoch, test
from resnet9 import ResNet9

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tiny')
    parser.add_argument('--target-epsilon', type=float, default=10)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale-norm', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm-layer', type=str, default='group')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--file-path', type=str, default=None)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=34)
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parse_args()

    SEED = args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)

    if args.file_path is None:
        file_path = './models/cifar10_resnet9_50_43.pth.tar'
    else:
        file_path = args.file_path

    checkpoint = torch.load(file_path)
    print('Parameters used for training')
    for key in checkpoint:
        if key not in ['state_dict', 'optimizer']:
            print(f'{key}: {checkpoint[key]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.layers == 1:
        layers = 'all'
    elif args.layers == 2:
        layers = 'conv4, res2, sn2, classifier'
    elif args.layers == 3:
        layers = 'classifier'
    elif args.layers == 4:
        layers = 'all with groups'

    config = dict(
        architecture='ResNet9',
        seed=SEED,
        epochs=args.epochs,
        dataset=args.dataset,
        batch_size=1024,
        privacy=args.privacy,
        target_epsilon=args.target_epsilon,
        scale_norm=args.scale_norm,
        norm_layer=args.norm_layer,
        num_groups=(32, 32, 32, 32),
        learning_rate=0.001,
        max_grad_norm=1.5,
        target_delta=1e-5,
        transfer_from=file_path,
        layers=layers,
        scheduler='ReduceLROnPlateau(patience=2, factor=0.5)',
        act_func=torch.nn.Mish
    )

    wandb.init(
        project='transfer-learning-cifar',
        entity='mrchntia',
        notes='analyze training of ResNet with and without ScaleNorm layers',
        config=config,
    )

    log_freq = 100

    model = ResNet9(
        3,
        10,
        act_func=config['act_func'],
        scale_norm=config['scale_norm'],
        norm_layer=config['norm_layer'],
        num_groups=config['num_groups']
    ).to(device)

    if config['dataset'] == 'tiny':
        num_classes = 200
        max_batch_size = 64
        train_loader, val_loader = get_tiny_dataloader(bs_train=config['batch_size'], bs_val=max_batch_size)
    elif config['dataset'] == 'cifar10':
        num_classes = 10
        max_batch_size = 521
        train_loader, val_loader = get_cifar_dataloader(
            bs_train=config['batch_size'],
            bs_val=max_batch_size,
            classes=num_classes,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    elif config['dataset'] == 'cifar100':
        num_classes = 100
        max_batch_size = 521
        train_loader, val_loader = get_cifar_dataloader(
            bs_train=config['batch_size'],
            bs_val=max_batch_size,
            classes=num_classes,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    if checkpoint['privacy']:
        state_dict = {}
        for key in checkpoint['state_dict'].keys():
            state_dict[key[8:]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict)
    state_dict = {}
    for key in checkpoint['state_dict'].keys():
        if key not in ['conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                       'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked',
                       'res1.0.1.running_mean', 'res1.0.1.running_var', 'res1.0.1.num_batches_tracked',
                       'res1.1.1.running_mean', 'res1.1.1.running_var', 'res1.1.1.num_batches_tracked',
                       'conv3.1.running_mean', 'conv3.1.running_var', 'conv3.1.num_batches_tracked',
                       'conv4.1.running_mean', 'conv4.1.running_var', 'conv4.1.num_batches_tracked',
                       'res2.0.1.running_mean', 'res2.0.1.running_var', 'res2.0.1.num_batches_tracked',
                       'res2.1.1.running_mean', 'res2.1.1.running_var', 'res2.1.1.num_batches_tracked',
                       'scale_norm_1.running_mean', 'scale_norm_1.running_var', 'scale_norm_1.num_batches_tracked',
                       'scale_norm_2.running_mean', 'scale_norm_2.running_var', 'scale_norm_2.num_batches_tracked']:
            state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict)

    if config['layers'] == 'conv4, res2, sn2, classifier':
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.res1.parameters():
            param.requires_grad = False
        for param in model.scale_norm_1.parameters():
            param.requires_grad = False
        for param in model.conv3.parameters():
            param.requires_grad = False
    elif config['layers'] == 'classifier':
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(1024, num_classes)
    model.to(device)

    if config['layers'] == 'all with groups':
        optimizer = torch.optim.NAdam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters(), 'lr': (config['learning_rate']/10)},
            {'params': model.res1.parameters(), 'lr': (config['learning_rate']/10)},
            {'params': model.scale_norm_1.parameters(), 'lr': (config['learning_rate']/10)},
            {'params': model.conv3.parameters(), 'lr': (config['learning_rate']/10)},
            {'params': model.conv4.parameters(), 'lr': (config['learning_rate'])},
            {'params': model.res2.parameters(), 'lr': (config['learning_rate'])},
            {'params': model.scale_norm_2.parameters(), 'lr': (config['learning_rate'])},
            {'params': model.classifier.parameters(), 'lr': config['learning_rate']}
        ], lr=(config['learning_rate']/10))
    else:
        optimizer = torch.optim.NAdam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    if config['privacy']:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_delta=config['target_delta'],
            target_epsilon=config['target_epsilon'],
            max_grad_norm=config['max_grad_norm'],
            epochs=config['epochs'],
        )
        print('privacy')

    wandb.watch(model, criterion, log='all', log_freq=log_freq)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # Train
    if config['privacy']:
        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_batch_size,
                optimizer=optimizer
        ) as new_train_loader:
            for epoch in range(config['epochs']):
                train_one_epoch(model, optimizer, criterion, new_train_loader, device, epoch, len(train_loader))
                test(model, val_loader, device)

    else:
        for epoch in range(config['epochs']):
            for batch_idx, (data, target) in enumerate(train_loader):
                train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, len(train_loader), lr_scheduler)
                test(model, val_loader, device)

    save_checkpoint(model, None, filename=f'models/transfer_learning_cifar_{wandb.run.id}.pth.tar')
    load_checkpoint(f'models/transfer_learning_cifar_{wandb.run.id}.pth.tar', model)

