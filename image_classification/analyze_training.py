import argparse
import random
import warnings
import numpy as np

import torch
import wandb
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from utils import get_cifar_dataloader, get_imagenette_dataloader, get_tiny_dataloader, test
from resnet9 import ResNet9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--target-epsilon', type=float, default=7.62)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale-norm', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm-layer', type=str, default='group')
    parser.add_argument('--epochs', type=int, default=25)
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    SEED = 34
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)

    args = parse_args()

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
    )

    wandb.init(
        project='analyze-scale-norm',
        entity='mrchntia',
        notes='analyze training of ResNet with and without ScaleNorm layers',
        config=config,
    )

    if config['dataset'] == 'cifar':
        num_classes = 10
        max_batch_size = 256
        train_loader, val_loader = get_cifar_dataloader(bs_train=config['batch_size'], bs_val=max_batch_size)
    elif config['dataset'] == 'imagenette':
        num_classes = 10
        max_batch_size = 32
        train_loader, val_loader = get_imagenette_dataloader(bs_train=config['batch_size'], bs_val=max_batch_size)
    elif config['dataset'] == 'tiny':
        num_classes = 200
        max_batch_size = 512
        train_loader, val_loader = get_tiny_dataloader(bs_train=config['batch_size'], bs_val=max_batch_size)
    else:
        raise ValueError(
            'Please specify a valid dataset.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet9(3, num_classes=num_classes, scale_norm=config['scale_norm'], norm_layer=config['norm_layer'],
                    num_groups=config['num_groups']).to(device)

    layers_names = {}
    log_count = 0
    log_freq = 48
    val = False

    def hook_fn(module, input, output):
        if log_count % log_freq == 0 and not val:
            wandb.log({f'activation_{layers_names[module]}': output})

    def get_all_layers(model):
        for name, layer in model._modules.items():
            if name not in ['FlatFeats', 'MP', 'classifier']:
                layers_names[layer] = name
                layer.register_forward_hook(hook_fn)


    get_all_layers(model)

    optimizer = torch.optim.NAdam(model.parameters(), lr=config['learning_rate'])

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
    criterion = torch.nn.CrossEntropyLoss()

    # modules_log = []
    # for name, layer in model._modules.items():
    #     modules_log.append(layer)

    wandb.watch(model, criterion, log='all', log_freq=log_freq)
    # Train
    if config['privacy']:
        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_batch_size,
                optimizer=optimizer
        ) as new_train_loader:
            for epoch in range(config['epochs']):
                model.train()
                for batch_idx, (data, target) in enumerate(new_train_loader):
                    log_count += 1
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    wandb.log({'epoch': epoch, 'loss': loss.item()})
                    if batch_idx % 20 == 0:
                        print(
                            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch,
                                batch_idx * len(data),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                            )
                        )

                # Test
                val = True
                test(model, val_loader, device)
                val = False
