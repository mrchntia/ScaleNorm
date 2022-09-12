import os
import subprocess
from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import optuna
import wandb


class MultiplePruners(optuna.pruners.BasePruner):
    def __init__(
            self,
            pruners: Iterable[optuna.pruners.BasePruner],
            pruning_condition: str = "any",
    ) -> None:

        self._pruners = tuple(pruners)

        self._pruning_condition_check_fn = None
        if pruning_condition == "any":
            self._pruning_condition_check_fn = any
        elif pruning_condition == "all":
            self._pruning_condition_check_fn = all
        else:
            raise ValueError(f"Invalid pruning ({pruning_condition}) condition passed!")
        assert self._pruning_condition_check_fn is not None

    def prune(
            self,
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial,
    ) -> bool:

        return self._pruning_condition_check_fn(pruner.prune(study, trial) for pruner in self._pruners)


class RepeatPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        # type: (optuna.Study, optuna.trial.FrozenTrial) -> bool

        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False


def save_checkpoint(model, params, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    state = {
        "state_dict": model.state_dict(),
    }
    if params is not None:
        for param in dir(params):
            if not param.startswith('__'):
                state[param] = getattr(params, param)
    torch.save(state, filename)


def load_checkpoint(file_path, model=None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(file_path)
    if model is not None:
        model.load_state_dict(checkpoint["state_dict"])
    print("Parameters used for training")
    for key in checkpoint:
        if key not in ["state_dict", "optimizer"]:
            print(f"{key}: {checkpoint[key]}")


def get_cifar_dataloader(bs_train=256, bs_val=256, classes=10, mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2471, 0.2435, 0.2616)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if classes == 10:
        train_ds = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        val_ds = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    if classes == 100:
        train_ds = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        val_ds = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )
    return train_loader, val_loader


def get_imagenette_dataloader(bs_train=64, bs_val=64, image_size=224):
    if not os.path.exists('./data/imagenette2-160/train'):
        print("path not existing")
        subprocess.run('wget -P ./data/ https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz', shell=True)
        subprocess.run('tar -xf ./data/imagenette2-320.tgz -C ./data/', shell=True)
    else:
        print("Dataset already downloaded.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_ds = datasets.ImageFolder(
        './data/imagenette2-160/train',
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
                transforms.Resize((image_size, image_size))
            ]
        )
    )
    val_ds = datasets.ImageFolder(
        './data/imagenette2-160/val',
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
                transforms.Resize((image_size, image_size))
            ]
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )
    return train_loader, val_loader


def get_tiny_dataloader(bs_train=64, bs_val=64):
    train_dir = "./data/tiny-imagenet-200/train"
    val_dir = "./data/tiny-imagenet-200/val"
    if not os.path.exists(train_dir):
        subprocess.run("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip", shell=True)
        subprocess.run("unzip -qq 'tiny-imagenet-200.zip'", shell=True)
        print("download done")
    else:
        print("Dataset already downloaded.")

    val_img_dir = os.path.join(val_dir, 'images')
    if len(os.listdir(val_img_dir)) != 200:
        # reorganize images in validation directory to fit expected folder structure of datasets.ImageFolder
        annotations_file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
        annotations = annotations_file.readlines()
        val_img_dict = {}
        for line in annotations:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        annotations_file.close()
        for img, folder in val_img_dict.items():
            new_path = (os.path.join(val_img_dir, folder))
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(new_path, img))
        print("reorganizing val folder done")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)

    train_ds = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomCrop(64, padding=4),
                normalize,
                transforms.RandomErasing(0.25),
            ]
        )
    )
    val_ds = datasets.ImageFolder(
        val_img_dir,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )
    return train_loader, val_loader


def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, epoch_len, lr_scheduler=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        wandb.log({"epoch": epoch, "loss": loss.item()})
        if batch_idx % 20 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / epoch_len,
                    loss.item(),
                )
            )
        if lr_scheduler:
            lr_scheduler.step(loss.item())


def test(model, val_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.CrossEntropyLoss(reduction="sum")(
                output, target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    wandb.log({"val_loss": test_loss, "val_acc": 100.0 * correct / len(val_loader.dataset)})
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
