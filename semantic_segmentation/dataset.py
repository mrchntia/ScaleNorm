import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # single channel image -> gray-scale
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = torch.unsqueeze(mask, dim=0)

        return image, mask


class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # single channel image -> gray-scale
        mask[mask == 0.0] = 0.0
        mask[mask != 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = torch.unsqueeze(mask, dim=0)

        return image, mask


class PascalDataset(Dataset):
    # TODO: not finished yet
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

        self.masks, self.labels = read_voc_images(voc_dir, is_train=is_train)
        self.maks = [self.normalize_image(mask) for mask in self.masks]
        self.colormap2label = voc_colormap2label()

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # load image as RGB image (3 channels) 
        image = np.array(Image.open(img_path).convert("RGB"))
        # load mask as gray-scale image (1 channel) -> each gray scale number represents one class
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # TODO: Maybe filtering for image size necessary
        # TODO: Convert gray-scale to numbers from 1-20

        return image, mask
