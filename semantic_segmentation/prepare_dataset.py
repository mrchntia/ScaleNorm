import subprocess

subprocess.run("mkdir ./data/pascalvoc2012/train_images", shell=True)
subprocess.run("mkdir ./data/pascalvoc2012/train_masks", shell=True)
subprocess.run("mkdir ./data/pascalvoc2012/val_images", shell=True)
subprocess.run("mkdir ./data/pascalvoc2012/val_masks", shell=True)

with open("../data/pascalvoc2012/VOC2012/ImageSets/Segmentation/train.txt") as train_set:
    for image_idx in train_set:
        subprocess.run(
            f"cp ./data/pascalvoc2012/VOC2012/JPEGImages/{image_idx.rstrip()}.jpg ./data/pascalvoc2012/train_images/",
            shell=True
        )
        subprocess.run(
            f"cp ./data/pascalvoc2012/VOC2012/SegmentationClass/{image_idx.rstrip()}.png ./data/pascalvoc2012/train_masks/",
            shell=True
        )

with open("../data/pascalvoc2012/VOC2012/ImageSets/Segmentation/val.txt") as val_set:
    for image_idx in val_set:
        subprocess.run(
            f"cp ./data/pascalvoc2012/VOC2012/JPEGImages/{image_idx.rstrip()}.jpg ./data/pascalvoc2012/val_images/",
            shell=True
        )
        subprocess.run(
            f"cp ./data/pascalvoc2012/VOC2012/SegmentationClass/{image_idx.rstrip()}.png ./data/pascalvoc2012/val_masks/",
            shell=True
        )
