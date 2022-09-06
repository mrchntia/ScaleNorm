import argparse
import random
import warnings
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np

import torch
from opacus import PrivacyEngine
from poutyne import Model, ReduceLROnPlateau, LambdaCallback

from utils import get_cifar_dataloader, get_imagenette_dataloader
from resnet9 import ResNet9
from resnet_pytorch import resnet18, resnet50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar', choices=['cifar', 'imagenette'])
    parser.add_argument('--model-arch', type=str, default='resnet9', choices=['resnet9', 'resnet18', 'resnet50'])
    parser.add_argument('--target-epsilon', type=float, default=7.62)
    parser.add_argument('--privacy', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale-norm', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm-layer', type=str, default='group')
    parser.add_argument('--epochs', type=int, default=25)
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

    if args.dataset == 'cifar':
        train_loader, val_loader = get_cifar_dataloader()
    elif args.dataset == 'imagenette':
        train_loader, val_loader = get_imagenette_dataloader()
    else:
        raise ValueError(
            "Please specify a valid dataset. ('cifar', 'imagenette')"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_arch == 'resnet9':
        model = ResNet9(3, 10, scale_norm=args.scale_norm, norm_layer=args.norm_layer, num_groups=(32, 32, 32, 32)
                        ).to(device)
        modules_to_visualize = [model.conv2, model.res1, model.scale_norm_1, model.conv4, model.res2,
                                model.scale_norm_2]
        plots_titles = [
            "conv2",
            "res1",
            "scale_norm_1",
            "conv4",
            "res2",
            "scale_norm_2"
        ]
    elif args.model_arch == 'resnet18':
        model = resnet18(num_groups=[32, 32, 32, 32], scale_norm=args.scale_norm).to(device)
        modules_to_visualize = [model.gn2, model.res1, model.scale_norm_1, model.conv4, model.res2,
                                model.scale_norm_2]
        plots_titles = [
            "conv2",
            "res1",
            "scale_norm_1",
            "conv4",
            "res2",
            "scale_norm_2"
        ]
    else:
        raise ValueError(
            "Please specify a valid model architecture."
        )

    for name, module in model.named_modules():
        print(name)

    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    visualisation = {}
    modules_to_visualize = [model.conv2, model.res1, model.scale_norm_1, model.conv4, model.res2, model.scale_norm_2]
    plots_titles = [
        "conv2",
        "res1",
        "scale_norm_1",
        "conv4",
        "res2",
        "scale_norm_2"
    ]
    get_all_layers(model)

    if args.privacy:
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

    learner = Model(model, optimizer, criterion, batch_metrics=["acc"], device=device)
    history = learner.fit_generator(train_loader, val_loader, epochs=args.epochs)

    rows = 2
    columns = 3
    fig = plt.figure(figsize=(columns * 6, rows * 4))
    fig.suptitle("test_title", fontsize=16)

    for i in range(len(modules_to_visualize)):
        fig.add_subplot(rows, columns, i+1)
        pickle.dump(
            visualisation[modules_to_visualize[i]],
            open("histograms/{}_{}_{}_s{}_p{}_resnet9_{}.p".format(
                args.dataset,
                args.norm_layer,
                args.scale_norm,
                args.privacy,
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
    plt.savefig('histograms/{}_resnet9_{}_{}_{}_s{}_p{}.png'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.dataset,
        args.epochs,
        args.norm_layer,
        args.scale_norm,
        args.privacy
    ))
    plt.close(fig)
    print("Plot saved")
    print(max([d["val_acc"] for d in history]))
