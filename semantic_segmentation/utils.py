import torch
import torchvision
from dataset import CarvanaDataset, MedicalDataset, PascalDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp


def save_checkpoint(model, params, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    state = {
        "state_dict": model.state_dict(),
    }
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


def get_carvana_dataloaders(params):
    train_dir = "../data/carvana/train_images/"
    train_mask_dir = "../data/carvana/train_masks/"
    val_dir = "../data/carvana/val_images/"
    val_mask_dir = "../data/carvana/val_masks/"
    pin_memory = True

    train_transform = A.Compose(
        [
            A.Resize(height=params.image_height, width=params.image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=params.image_height, width=params.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_pancreas_dataloaders(params):
    train_dir = "/media/HDD/MSD/Pancreas_PNGs/MSD_Image_PNGs"
    train_mask_dir = "/media/HDD/MSD/Pancreas_PNGs/MSD_Mask_PNGs"
    val_dir = "/media/HDD/MSD/Pancreas_PNGs_Val/Images"
    val_mask_dir = "/media/HDD/MSD/Pancreas_PNGs_Val/Masks"
    pin_memory = True

    train_transform = A.Compose([
        A.Resize(height=params.image_height, width=params.image_width),
        A.Rotate(limit=10, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])  # TODO: Maybe RandomCrop or resize
    val_transform = A.Compose([
        A.Resize(height=params.image_height, width=params.image_width),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])  # TODO: Maybe RandomCrop or resize

    train_ds = MedicalDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_ds = MedicalDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def get_liver_dataloaders(params):
    train_dir = "/media/HDD/MSD/Liver/train/images_jpg"
    train_mask_dir = "/media/HDD/MSD/Liver/train/masks_jpg"
    val_dir = "/media/HDD/MSD/Liver/val/images_jpg"
    val_mask_dir = "/media/HDD/MSD/Liver/val/masks_jpg"
    pin_memory = True

    train_transform = A.Compose([
        A.Resize(height=params.image_height, width=params.image_width),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=params.image_height, width=params.image_width),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2()
    ])

    train_ds = MedicalDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_ds = MedicalDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def get_pascal_dataloaders(params):
    train_dir = "../data/pascalvoc2012/train_images/"
    train_mask_dir = "../data/pascalvoc2012/train_masks/"
    val_dir = "../data/pascalvoc2012/val_images/"
    val_mask_dir = "../data/pascalvoc2012/val_masks/"
    pin_memory = True

    train_transform = A.Compose(
        [
            # TODO: Maybe RandomCrop is better than resize
            A.Resize(height=params.image_height, width=params.image_width),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=params.image_height, width=params.image_width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_ds = PascalDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_ds = PascalDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def train(loader, model, optimizer, loss_fn, device):
    loop = tqdm(loader)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)
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


def test(loader, model, loss_fn, device):
    dice_scores = []
    loss_values = []
    threshold_value = 0.5
    model.eval()

    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            predictions = torch.sigmoid(model(image))
            predictions_n = predictions.detach().cpu().numpy().squeeze()
            masks_n = mask.detach().cpu().numpy().squeeze()
            threshold = lambda x: np.where(x > threshold_value, 1.0, 0.0)
            dice = 1 - smp.utils.losses.DiceLoss()(
                torch.from_numpy(threshold(predictions_n)), torch.from_numpy(masks_n)
            )
            dice_scores.append(dice)

            loss_values.append(loss_fn(predictions, mask))

    loss = torch.mean(torch.stack(loss_values)).item()
    print(f"Average val-loss: {loss:.2f}")
    dice_score = torch.mean(torch.stack(dice_scores)).item()
    print(f"Dice Score @ threshold {threshold_value}: {dice_score:.2f}")
    return dice_score, loss


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
