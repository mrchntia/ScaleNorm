from typing import Dict, Callable, Tuple
import argparse
import random
import numpy as np
import warnings

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import optuna
from optuna import TrialPruned
from poutyne import LambdaCallback, ReduceLROnPlateau, Model

from resnet9 import ResNet9
from resnet_pytorch import resnet50
from utils import get_cifar_dataloader, get_imagenette_dataloader, save_checkpoint, load_checkpoint, get_tiny_dataloader
from utils import RepeatPruner, MultiplePruners

act_funcs: Dict[str, Callable] = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'mish': nn.Mish,
}


class Parameters:
    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 act_func: str,
                 model_arch: str,
                 epochs: int,
                 target_epsilon: float,
                 grad_norm: float,
                 noise_mult: float,
                 privacy: bool,
                 scale_norm: bool,
                 norm_layer: str,
                 num_groups: Tuple[int, ...]
                 ):
        self.dataset: str = dataset
        self.epochs: int = epochs
        self.batch_size: float = batch_size
        self.target_epsilon: float = target_epsilon
        self.grad_norm: float = grad_norm
        self.noise_mult: float = noise_mult
        self.model_arch: str = model_arch
        self.privacy: bool = privacy
        self.act_func: Callable = act_funcs[act_func]
        self.scale_norm: bool = scale_norm
        self.norm_layer: str = norm_layer
        self.num_groups: Tuple[int, ...] = num_groups

        # Fixed
        if self.dataset == 'cifar':
            self.max_batch_size = 256
            self.image_size = None
            self.in_channels: int = 3
            self.num_classes: int = 10
        if self.dataset == 'imagenette':
            self.max_batch_size = 32
            self.image_size = 160
            self.in_channels: int = 3
            self.num_classes: int = 10
        if self.dataset == 'tiny':
            self.max_batch_size = 521
            self.image_size = None
            self.in_channels: int = 3
            self.num_classes: int = 200
        self.learning_rate: float = 0.001
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_delta: float = 1e-5
        self.log_interval: int = 20
        self.secure_rng: bool = False
        self.group_norm: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar', choices=['cifar', 'imagenette', 'tiny'])
    parser.add_argument('--model-arch', type=str, default='resnet9', choices=['resnet9', 'resnet18', 'resnet50'])
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--act-func', type=str, default='relu', choices=['tanh', 'relu', 'mish'])
    parser.add_argument('--target-epsilon', type=float, default=None)
    parser.add_argument('--noise-mult', type=float, default=None)
    parser.add_argument('--grad-norm', type=float, default=1.5)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale-norm', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm-layer', type=str, default='group')
    parser.add_argument('--num-groups', type=Tuple[int, ...], default=(32, 32, 32, 32))
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parse_args()
    params = Parameters(
        args.dataset,
        args.batch_size,
        args.act_func,
        args.model_arch,
        args.epochs,
        args.target_epsilon,
        args.grad_norm,
        args.noise_mult,
        args.privacy,
        args.scale_norm,
        args.norm_layer,
        args.num_groups
    )

    def objective(trial):
        params.target_epsilon = trial.suggest_categorical('target_epsilon', [3, 10, 7.62])
        num_groups = trial.suggest_categorical('num_groups', [1, 8, 16, 32, 64, 2048])
        params.num_groups = (num_groups, num_groups, num_groups, num_groups)
        params.norm_layer = trial.suggest_categorical('norm_layer', ['batch', 'group'])
        params.scale_norm = trial.suggest_categorical('scale_norm', [True, False])
        seed = trial.suggest_categorical('seed', [50, 34, 113])
        act_func = trial.suggest_categorical('act_func', ['relu', 'mish'])
        params.act_func = act_funcs[act_func]
        params.epochs = trial.suggest_categorical('epochs', [25, 50, 90])

        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if params.model_arch == 'resnet9':
            model = ResNet9(
                params.in_channels,
                params.num_classes,
                params.act_func,
                params.scale_norm,
                params.norm_layer,
                params.num_groups
            ).to(params.device)
        if params.model_arch == 'resnet50':
            model = resnet50(
                num_groups=params.num_groups,
                scale_norm=params.scale_norm,
                num_classes=params.num_classes,
                act_func=params.act_func
            ).to(params.device)
        # total_params = sum(p.numel() for p in model.parameters())
        if params.dataset == 'cifar':
            train_loader, val_loader = get_cifar_dataloader(bs_train=params.batch_size, bs_val=params.max_batch_size)
        elif params.dataset == 'imagenette':
            train_loader, val_loader = get_imagenette_dataloader(
                bs_train=params.batch_size,
                bs_val=params.max_batch_size,
                image_size=params.image_size
            )
        elif params.dataset == 'tiny':
            train_loader, val_loader = get_tiny_dataloader(bs_train=params.batch_size, bs_val=params.max_batch_size)
        else:
            raise ValueError(
                "Please specify a valid dataset. ('cifar', 'imagenette', 'tiny')"
            )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(model.parameters(), params.learning_rate)

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
                    max_grad_norm=params.grad_norm,
                    noise_generator=torch.Generator(device=params.device).manual_seed(seed)
                )
            else:
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=params.noise_mult,
                    max_grad_norm=params.grad_norm,
                    noise_generator=torch.Generator(device=params.device).manual_seed(seed)
                )

        print('Training starts')

        def report_prune(logs, epoch_number):
            trial.report(logs['val_acc'], epoch_number)
            if trial.should_prune():
                raise TrialPruned()

        report_prune_cb = LambdaCallback(on_epoch_end=lambda epoch_number, logs: report_prune(logs, epoch_number))
        lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, verbose=True)
        # lr_scheduler = OneCycleLR(max_lr=0.02, steps_per_epoch=len(train_loader),
        #                           epochs=params.epochs, div_factor=10, final_div_factor=10,
        #                           pct_start=10 / params.epochs, verbose=True)
        cbs = [report_prune_cb, lr_scheduler]
        learner = Model(model, optimizer, criterion, batch_metrics=['acc'], device=params.device)

        print(len(train_loader))

        if params.privacy:
            with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=params.max_batch_size,
                    optimizer=optimizer
            ) as new_train_loader:
                history = learner.fit_generator(new_train_loader, val_loader, epochs=params.epochs, callbacks=cbs)
        else:
            history = learner.fit_generator(train_loader, val_loader, epochs=params.epochs, callbacks=cbs)

        if params.privacy:
            params.final_epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=params.target_delta)

        save_checkpoint(model, params, filename=f'models/{study_name}_{trial.number}.pth.tar')
        load_checkpoint(f'models/{study_name}_{trial.number}.pth.tar', model)

        return max([d['val_acc'] for d in history])

    study_name = 'cifar_epochs'
    storage = 'sqlite:///scaleresnet.db'

    # if study_name == 'test_code':
    #     optuna.delete_study(study_name, storage)
    search_space = {
        'target_epsilon': [7.62],  # 3, 7.62, 10 # 0, 10, 100, 1000
        'num_groups': [32],  # 1, 8, 16, 32, 64, 2048
        'norm_layer': ['group'],
        'scale_norm': [False],
        'seed': [34, 50, 113],  # 34, 50, 113
        'act_func': ['mish'],
        'epochs': [25]
    }
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.GridSampler(search_space),
        # pruner=MultiplePruners((
        #     optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10),
        #     RepeatPruner()
        # )),
        # pruner=RepeatPruner(),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)

