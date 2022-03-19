import torch
import torch.nn as nn
from opacus import PrivacyEngine
import optuna
from optuna import TrialPruned

import argparse
from tqdm import tqdm
from typing import Tuple
import warnings
import numpy as np
import random

from model import UNET
from monet import MoNet
from parameter import Parameters
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_carvana_dataloaders,
    check_accuracy,
)


def train_fn(loader, model, optimizer, loss_fn, device):
    loop = tqdm(loader)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        optimizer.zero_grad()

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="carvana", choices=["carvana"])
    parser.add_argument("--model-arch", type=str, default="monet", choices=["monet"])
    parser.add_argument("--batch-size", type=int, default=1024)
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
        params.num_groups = trial.suggest_categorical("num_groups", [1, 8, 16, 32, 64, 2048])
        params.scale_norm = trial.suggest_categorical("scale_norm", [True, False])
        params.seed = trial.suggest_categorical("seed", [788, 374, 39])

        torch.backends.cudnn.deterministic = True
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)

        # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        if params.model_arch == "monet":
            model = MoNet(
                in_channels=params.in_channels,
                out_channels=params.out_channels,
                scale_norm=params.scale_norm,
                activation=params.act_func
            ).to(params.device)

        print(model)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = params.optimizer(model.parameters(), lr=params.learning_rate)

        train_loader, val_loader = get_carvana_dataloaders(params)

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

        # load_checkpoint("my_checkpoint.pth.tar", model, optimizer)
        # check_accuracy(val_loader, model, params.device)
        val_acc_list = []
        for epoch in range(params.epochs):
            train_fn(train_loader, model, optimizer, loss_fn, params.device)
            val_acc = check_accuracy(val_loader, model, params.device)
            val_acc_list.append(val_acc)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise TrialPruned()

        params.val_acc_list = val_acc_list

        if params.privacy:
            params.final_epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=params.target_delta)

        save_checkpoint(model, optimizer, params, filename=f"models/{study_name}_{trial.number}.pth.tar")

        return max(val_acc_list)

    study_name = "monet_scale_groups_dp_10"
    storage = "sqlite:///segmentation.db"
    # optuna.delete_study(study_name, storage)
    search_space = {
        "num_groups": [1, 8, 16, 32, 64],  # 16, 32, 64, 2048
        "scale_norm": [True, False],
        "seed": [788, 374, 39],
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
        # pruner=RepeatPruner(),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=12)


if __name__ == "__main__":
    main()
