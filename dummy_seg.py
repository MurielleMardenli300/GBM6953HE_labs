import torch
import json
import torch.nn.functional as F
from unet_mri_utils import UNetModule  # your module
from monai.metrics import DiceMetric
import numpy as np

from unet_mri_utils import BrainMRIDataset, MergeSegLabels, AddChannelDim
from monai.transforms import Compose, ResizeD, ScaleIntensityRanged, EnsureTyped
from torch.utils.data import DataLoader

from monai.networks.utils import one_hot

type = "4labels"

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
    
model = UNetModule(
    learning_rate=1e-4,
    num_classes=num_classes
)

# best_model_path = "/home/boadem/Work/School/Unet_brain_mri_models_WGM/best_model_fold_4.pth"
best_model_path = f"/home/boadem/Work/School/Unet_brain_mri_models_{type}/best_model_fold_1.pth"

state_dict = torch.load(best_model_path, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

device = torch.device("cpu")
model.to(device)

print("Model loaded and ready for inference.")

test_set = json.load(open(test_set_path))



test_transforms = Compose([
    AddChannelDim(keys=["image", "label"]),
    MergeSegLabels(keys=["label"], labels_file_path=labels_file_path, type=type),
    ResizeD(keys=["image", "label"], spatial_size=(128,128), mode=("bilinear", "nearest")),
    # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
])

test_dataset = BrainMRIDataset(
            img_paths=[item["image"] for item in test_set],
            seg_paths=[item["segmentation"] for item in test_set],
            transform=test_transforms
        )

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
dice_scores = []
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

all_batch_dice = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device) # Shape: [B, 1, H, W]

        outputs = model(images) # Shape: [B, num_classes, H, W]
        
        # 1. Get the class index per pixel [B, 1, H, W]
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        
        # 2. Use MONAI's one_hot to get [B, num_classes, H, W]
        # This ensures the Class dimension is at index 1
        outputs_onehot = one_hot(preds, num_classes=num_classes)
        segs_onehot = one_hot(labels, num_classes=num_classes)

        # 3. Calculate dice (with reduction="mean_batch", this returns a tensor of size [num_classes])
        batch_dice = dice_metric(y_pred=outputs_onehot, y=segs_onehot) 
        
        # Log batch progress
        # print(f"Batch {batch_idx+1} processed.")

        print(batch_dice)

# 2. Aggregate Results (This will now correctly be an array of length num_classes)
mean_dice_per_class = dice_metric.aggregate().cpu().numpy()
overall_average_dice = np.mean(mean_dice_per_class)
