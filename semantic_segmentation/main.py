import torch
import torch.nn as nn
from opacus import PrivacyEngine
import optuna
from optuna import TrialPruned
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
from typing import Tuple
import warnings
import numpy as np
import random

from unet9 import UNet9
from monet import MoNet
from parameter import Parameters
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_carvana_dataloaders,
    get_pancreas_dataloaders,
    get_liver_dataloaders,
    test,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="carvana", choices=["carvana", "pancreas", "liver"])
    parser.add_argument("--model-arch", type=str, default="monet", choices=["monet", "unet", "unet9"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--act-func", type=str, default="mish", choices=["tanh", "relu", "mish"])
    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument("--noise-mult", type=float, default=None)
    parser.add_argument("--grad-norm", type=float, default=1.5)
    parser.add_argument("--privacy", type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-norm", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-layer", type=str, default="group")
    parser.add_argument("--num-groups", type=Tuple[int, ...], default=(32, 32, 32, 32))
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")

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
        params.privacy = trial.suggest_categorical("privacy", [True, False])
        params.target_epsilon = trial.suggest_categorical("target_epsilon", [10, 5, 3, 0])
        params.norm_layer = trial.suggest_categorical("norm_layer", ["group", "batch"])
        params.num_groups = trial.suggest_categorical("num_groups", [1, 8, 16, 32, 64, 2048, 0])
        params.scale_norm = trial.suggest_categorical("scale_norm", [True, False])
        params.seed = trial.suggest_categorical("seed", [788, 374, 39])

        torch.backends.cudnn.deterministic = True
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)

        if params.model_arch == "unet":
            model = smp.Unet(
                encoder_weights=None,
                in_channels=params.in_channels,
                classes=params.out_channels
            ).to(params.device)
        elif params.model_arch == "monet":
            model = MoNet(
                in_channels=params.in_channels,
                out_channels=params.out_channels,
                scale_norm=params.scale_norm,
                activation=params.act_func,
                norm=params.norm_layer
            ).to(params.device)
        elif params.model_arch == "unet9":
            model = UNet9(
                in_channels=params.in_channels,
                num_classes=params.out_channels,
                scale_norm=params.scale_norm,
                act_func=params.act_func,
                norm_layer=params.norm_layer,
                num_groups=params.num_groups,
            ).to(params.device)
        # print(model)

        # criterion = nn.BCEWithLogitsLoss()
        criterion = smp.losses.DiceLoss(mode="binary")
        optimizer = params.optimizer(model.parameters(), lr=params.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, verbose=True)

        if params.dataset == "carvana":
            train_loader, val_loader = get_carvana_dataloaders(params)
        elif params.dataset == "pancreas":
            train_loader, val_loader = get_pancreas_dataloaders(params)
        elif params.dataset == "liver":
            train_loader, val_loader = get_liver_dataloaders(params)

        if params.privacy:
            privacy_engine = PrivacyEngine()
            if params.target_epsilon:
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=params.target_epsilon,
                    target_delta=1e-5,
                    epochs=params.epochs,
                    max_grad_norm=1.5,
                    noise_generator=torch.Generator(device=params.device).manual_seed(params.seed)
                )

        # load_checkpoint("models/liver_monet_scale_groups_dp_10_dice_score_8.pth.tar", model)
        # test(val_loader, model, params.device)

        val_loss_list = []
        dice_score_list = []
        for epoch in range(params.epochs):
            train(train_loader, model, optimizer, criterion, params.device)
            dice_score, loss = test(val_loader, model, criterion, params.device)
            val_loss_list.append(loss)
            dice_score_list.append(dice_score)  # .cpu().detach().numpy()
            trial.report(dice_score, epoch)
            if trial.should_prune():
                raise TrialPruned()
            scheduler.step(dice_score)

        params.val_loss_list = val_loss_list
        params.dice_score_list = dice_score_list

        if params.privacy:
            params.final_epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=params.target_delta)

        save_checkpoint(model, params, filename=f"models/{study_name}_{trial.number}.pth.tar")
        # load_checkpoint(f"models/{study_name}_{trial.number}.pth.tar", model)

        return max(dice_score_list)

    study_name = "liver_unet9"
    # study_name = "test_code"
    storage = "sqlite:///segmentation.db"
    # optuna.delete_study(study_name, storage)
    search_space = {
        "privacy": [True],  # True, False
        "target_epsilon": [10],  # 10, 5, 3, 0
        "norm_layer": ["group"],
        "num_groups": [1, 8, 16, 32, 64, 2048],  # 0, 1, 8, 16, 32, 64, 2048
        "scale_norm": [True, False],  # True, False
        "seed": [788, 374, 39],  # 788, 374, 39
    }
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.GridSampler(search_space),
        # pruner=MultiplePruners((
        #     optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10),
        #     RepeatPruner()
        # )),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=36)


if __name__ == "__main__":
    main()
