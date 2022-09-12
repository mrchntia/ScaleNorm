import argparse
import random
import warnings

import numpy as np
import poutyne

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from poutyne import Model

from utils import get_tiny_dataloader, get_cifar_dataloader, save_checkpoint, load_checkpoint, train_one_epoch, test
from resnet9 import ResNet9

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tiny')
    parser.add_argument('--target-epsilon', type=float, default=0)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale-norm', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm-layer', type=str, default='group')
    parser.add_argument('--epochs', type=int, default=90)
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    SEED = 34
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = dict(
        architecture="ResNet9",
        seed=SEED,
        epochs=args.epochs,
        dataset=args.dataset,
        batch_size=521,
        privacy=args.privacy,
        target_epsilon=args.target_epsilon,
        scale_norm=args.scale_norm,
        norm_layer=args.norm_layer,
        num_groups=(32, 32, 32, 32),
        learning_rate=0.001,
        max_grad_norm=1.5,
        target_delta=1e-5,
        scheduler="ReduceLROnPlateau(patience=2, factor=0.5)",
        act_func=torch.nn.Mish
    )

    wandb.init(
        project="transfer-learning-cifar",
        entity="mrchntia",
        notes="analyze training of ResNet with and without ScaleNorm layers",
        config=config,
    )

    log_freq = 1000

    if config["dataset"] == 'tiny':
        num_classes = 200
        max_batch_size = 64
        train_loader, val_loader = get_tiny_dataloader(bs_train=config["batch_size"], bs_val=max_batch_size)
    elif config["dataset"] == 'cifar10':
        num_classes = 10
        max_batch_size = 521
        train_loader, val_loader = get_cifar_dataloader(
            bs_train=config["batch_size"],
            bs_val=max_batch_size,
            classes=num_classes,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    elif config["dataset"] == 'cifar100':
        num_classes = 100
        max_batch_size = 521
        train_loader, val_loader = get_cifar_dataloader(
            bs_train=config["batch_size"],
            bs_val=max_batch_size,
            classes=num_classes,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )

    model = ResNet9(
        3,
        num_classes=num_classes,
        act_func=config["act_func"],
        scale_norm=config["scale_norm"],
        norm_layer=config["norm_layer"],
        num_groups=config["num_groups"]
    ).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    if config["privacy"]:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_delta=config["target_delta"],
            target_epsilon=config["target_epsilon"],
            max_grad_norm=config["max_grad_norm"],
            epochs=config["epochs"],
        )
        print("privacy")

    wandb.watch(model, criterion, log="all", log_freq=log_freq)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

    # Train
    if config["privacy"]:
        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_batch_size,
                optimizer=optimizer
        ) as new_train_loader:
            for epoch in range(config["epochs"]):
                train_one_epoch(
                    model,
                    optimizer,
                    criterion,
                    train_loader,
                    device,
                    epoch,
                    len(train_loader),
                    lr_scheduler
                )
                test(model, val_loader, device)

    else:
        for epoch in range(config["epochs"]):
            train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, len(train_loader), lr_scheduler)
            test(model, val_loader, device)

    save_checkpoint(model, None, filename=f"models/transfer_learning_cifar_{wandb.run.id}.pth.tar")
    load_checkpoint(f"models/transfer_learning_cifar_{wandb.run.id}.pth.tar", model)

