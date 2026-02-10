# UNet Implementation with MONAI DiceMetric
# Slices are assumed aligned in template space

import os
import json
import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    MapTransform,
    ResizeD,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotateD,
    RandGaussianNoised,
    EnsureTyped,
)

from sklearn.model_selection import KFold



def get_tissue_ids(labels_file_path, type="WGM"):
    """ Get tissue Ids.
    Args:
        labels_file_path (str): Path to the labels text file.
        type (str): Type of tissue IDs to extract. Options are "WGM" for white and gray matter, 
                    "4labels" for Cortex, Subcortical GM structures, WM, CSF
        Returns:
            if type=="WGM":
                gm_ids (list): List of gray matter label IDs.
                wm_ids (list): List of white matter label IDs.
            elif type=="4labels":
                list of the labels
    """
    if type == "WGM":
        gm_ids, wm_ids = [], []
        with open(labels_file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                label_id, label_name = int(parts[0]), parts[1].lower()
                if any(x in label_name for x in ['cortex', 'thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens']):
                    gm_ids.append(label_id)
                elif "white-matter" in label_name:
                    wm_ids.append(label_id)
        print(f"Found GM IDs: {gm_ids}")
        print(f"Found WM IDs: {wm_ids}")
        return gm_ids, wm_ids
    
    elif type == "4labels":
        label_ids = []
        with open(labels_file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                label_id = int(parts[0])
                label_ids.append(label_id)
        print(f"Found label IDs: {label_ids}")
        return label_ids




class BrainMRIDataset(Dataset):
    def __init__(self, img_paths, seg_paths, transform=None):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx]).get_fdata().squeeze()
        seg = nib.load(self.seg_paths[idx]).get_fdata().squeeze()

        #Images are initially cropped to remove excess background for less noise during training
        img = img[:, :160] 
        seg = seg[:, :160]

        sample = {"image": img.astype(np.float32), "label": seg.astype(np.int64)}
        if self.transform:
            sample = self.transform(sample)
        return sample



class MergeSegLabels(MapTransform):
    """ Merging segmentation labels based on the content of the labels _file and the type of segmentation(white, gray matter, or 4 labels)"""

    def __init__(self, keys, labels_file_path, type="WGM"):
        super().__init__(keys)
        if type == "WGM":
            self.gm_ids, self.wm_ids = get_tissue_ids(labels_file_path, type=type)
            self.ids = []
        elif type == "4labels":
            self.ids = get_tissue_ids(labels_file_path, type=type)
    

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            new_seg = np.zeros_like(seg)
            if hasattr(self, 'gm_ids') and hasattr(self, 'wm_ids'):
                new_seg[np.isin(seg, self.gm_ids)] = 1
                new_seg[np.isin(seg, self.wm_ids)] = 2
            elif hasattr(self, 'ids'):
                for i, label_id in enumerate(self.ids):
                    new_seg[seg == label_id] = i
            d[key] = new_seg
        return d



class AddChannelDim(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if d[key].ndim == 2:
                d[key] = np.expand_dims(d[key], axis=0)
        return d



class UNetModule(pl.LightningModule):
    """
    Docstring for UNetModule
    num_classes :  number of classes for segmentation (including background)
    """
    def __init__(self, learning_rate=1e-3, num_classes=3):
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, segs = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_function(outputs, segs)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, segs = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_function(outputs, segs)

        #We convert the segmentations to one hot for dice computation
        outputs_onehot = F.one_hot(torch.argmax(outputs, dim=1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        segs_onehot = F.one_hot(segs.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        self.dice_metric(y_pred=outputs_onehot, y=segs_onehot)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", dice_score, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

type = "WGM"

if type == "WGM":
    num_classes = 3
    labels_file_path = "/home/boadem/Work/School/neurite_data/seg24_labels.txt"
    train_set_path = "/home/boadem/Work/School/train_set_paths.json"
    test_set_path = "/home/boadem/Work/School/test_set_paths.json"
elif type == "4labels":
    num_classes = 5
    labels_file_path = "/home/boadem/Work/School/neurite_data/seg4_labels.txt"
    train_set_path = "/home/boadem/Work/School/train_set_paths_4labels.json"
    test_set_path = "/home/boadem/Work/School/test_set_paths_4labels.json"

train_transforms = Compose([
    AddChannelDim(keys=["image", "label"]),
    MergeSegLabels(keys=["label"], labels_file_path=labels_file_path, type=type),
    ResizeD(keys=["image", "label"], spatial_size=(128,128), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
    RandRotateD(keys=["image", "label"], range_x=15, prob=0.3, mode=("bilinear", "nearest")),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
])

val_transforms = Compose([
    AddChannelDim(keys=["image", "label"]),
    MergeSegLabels(keys=["label"], labels_file_path=labels_file_path, type=type),
    ResizeD(keys=["image", "label"], spatial_size=(128,128), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
])



def train(data, save_dir, log_save_dir, num_classes, n_epochs=60, patience=10):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42) #Cross validation with 5 splits

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\nStarting fold {fold+1}...")

        train_paths = [data[i] for i in train_idx]
        val_paths = [data[i] for i in val_idx]

        train_dataset = BrainMRIDataset(
            img_paths=[item["image"] for item in train_paths],
            seg_paths=[item["segmentation"] for item in train_paths],
            transform=train_transforms
        )
        
        val_dataset = BrainMRIDataset(
            img_paths=[item["image"] for item in val_paths],
            seg_paths=[item["segmentation"] for item in val_paths],
            transform=val_transforms
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

        model = UNetModule(num_classes=num_classes,learning_rate=1e-4)

        early_stopping = pl.callbacks.EarlyStopping(monitor="val_dice", patience=patience, mode="max") #We monitor the validation dice for early stopping
        logger = pl.loggers.TensorBoardLogger(log_save_dir, name=f"unet_fold_{fold+1}")

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[early_stopping],
            logger=logger,
            log_every_n_steps=10,
            accelerator="auto",
            devices=1
        )

        trainer.fit(model, train_loader, val_loader)

        model_path = os.path.join(save_dir, f"best_model_fold_{fold+1}.pth")
        trainer.save_checkpoint(model_path)
        torch.save(model.state_dict(), model_path)
        print(f"Best model for fold {fold+1} saved at: {model_path}")

    print("Training complete for all folds.")



train_set = json.load(open(train_set_path))

train(
    data=train_set,
    save_dir=f"/home/boadem/Work/School/Unet_brain_mri_models_{type}",
    log_save_dir=f"/home/boadem/Work/School/Unet_brain_mri_logs_{type}",
    num_classes=num_classes,
    n_epochs=100,
    patience=10
)
