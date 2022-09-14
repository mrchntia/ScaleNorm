import argparse
import random
import warnings
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (
    get_liver_dataloaders,
    test,
    train
)
from linknet9 import LinkNet9
from unet9 import UNet9
from parameter import Parameters

import segmentation_models_pytorch as smp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="carvana", choices=["carvana", "pancreas", "liver"])
    parser.add_argument("--model-arch", type=str, default="monet", choices=["monet", "unet", "unet9", "linknet9"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--act-func", type=str, default="mish", choices=["tanh", "relu", "mish"])
    parser.add_argument("--target-epsilon", type=float, default=3)
    parser.add_argument("--noise-mult", type=float, default=None)
    parser.add_argument("--grad-norm", type=float, default=1.5)
    parser.add_argument("--privacy", type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-norm", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-layer", type=str, default="group")
    parser.add_argument("--num-groups", type=int, default=32)
    return parser.parse_args()


def hook_fn(module, input, output):
    visualisation[module] = np.transpose(torch.flatten(output.cpu()).detach().numpy())


def get_all_layers(model):
    for name, layer in model._modules.items():
        layer.register_forward_hook(hook_fn)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    SEED = 1302
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)

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

    if args.dataset == 'liver':
        train_loader, val_loader = get_liver_dataloaders(params)
    else:
        raise ValueError(
            "Please specify a valid dataset. ('liver')"
        )

    if args.model_arch == 'linknet9':
        model = LinkNet9(1, 1, scale_norm=args.scale_norm, norm_layer=args.norm_layer).to(params.device)
    elif args.model_arch == 'unet9':
        model = UNet9(1, 1, scale_norm=args.scale_norm, norm_layer=args.norm_layer).to(params.device)
    else:
        raise ValueError(
            "Please specify a valid architecture. ('linknet9', 'unet9')"
        )
    modules_to_visualize = [model.conv2, model.res1, model.scale_norm_1,
                            model.conv4, model.res2, model.scale_norm_2,
                            model.conv5, model.deco1, model.scale_norm_3,
                            model.scale_norm_3, model.res3, model.scale_norm_4,
                            model.deco3, model.deco4, model.scale_norm_5,
                            model.scale_norm_5, model.res4, model.scale_norm_6]
    plots_titles = [
        "conv2", "res1", "scale_norm_1",
        "conv4", "res2", "scale_norm_2",
        "conv5", "deco1", "scale_norm_3",
        "scale_norm_3", "res3", "scale_norm_4",
        "deco3", "deco4", "scale_norm_5",
        "scale_norm_5", "res4", "scale_norm_6"
    ]

    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5, verbose=True)

    visualisation = {}
    get_all_layers(model)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_delta=1e-5,
        target_epsilon=args.target_epsilon,
        max_grad_norm=1.5,
        epochs=args.epochs,
    )
    optimizer.defaults = optimizer.original_optimizer.defaults

    val_loss_list = []
    dice_score_list = []
    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=16,
            optimizer=optimizer
    ) as memory_safe_data_loader:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}:")
            train(memory_safe_data_loader, model, optimizer, criterion, params.device)
            dice_score, loss = test(val_loader, model, criterion, params.device)
            val_loss_list.append(loss)
            dice_score_list.append(dice_score)  # .cpu().detach().numpy()
            scheduler.step(dice_score)

    print(f"acc: {max(dice_score_list)}, final eps: {privacy_engine.accountant.get_privacy_spent(delta=1e-5)}")

    rows = 6
    columns = 3
    fig = plt.figure(figsize=(columns * 6, rows * 4))
    fig.suptitle("test_title", fontsize=16)

    for i in range(len(modules_to_visualize)):
        fig.add_subplot(rows, columns, i+1)
        pickle.dump(
            visualisation[modules_to_visualize[i]],
            open("histograms/{}_{}_{}_s{}_p{}_{}_{}.p".format(
                args.dataset,
                args.epochs,
                args.norm_layer,
                args.scale_norm,
                args.privacy,
                args.model_arch,
                plots_titles[i]
            ), "wb")
        )
        plt.hist(
            visualisation[modules_to_visualize[i]],
            bins=50,
            range=(-2, 6)
        )
        plt.title("{}".format(plots_titles[i]))
    fig.gca().autoscale()
    plt.savefig('histograms/{}_{}_{}_{}_{}_s{}_p{}.png'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.model_arch,
        args.dataset,
        args.epochs,
        args.norm_layer,
        args.scale_norm,
        args.privacy
    ))
    plt.close(fig)
    print("Plot saved")
