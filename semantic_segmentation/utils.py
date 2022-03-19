import torch
import torchvision
from dataset import CarvanaDataset, PascalDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def save_checkpoint(model, optimizer, params, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    for param in dir(params):
        if not param.startswith('__'):
            state[param] = getattr(params, param)
    torch.save(state, filename)


def load_checkpoint(file_path, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Parameters used for training")
    for key in checkpoint:
        if key not in ["state_dict", "optimizer"]:
            print(f"{key}: {checkpoint[key]}")


def get_carvana_dataloaders(params):
    train_dir = "..data/carvana/train_images/"
    train_mask_dir = "..data/carvana/train_masks/"
    val_dir = "..data/carvana/val_images/"
    val_mask_dir = "..data/carvana/val_masks/"
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

    train_ds = PascalDataset(
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

    val_ds = PascalDataset(
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


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    acc = num_correct/num_pixels*100

    print(
        f"Got {num_correct}/{num_pixels} with acc {acc:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    return acc


# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
