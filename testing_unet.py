import torch
import json
import torch.nn.functional as F
from unet_mri_utils import UNetModule  # your module
from monai.metrics import DiceMetric
import numpy as np

model = UNetModule(
    learning_rate=1e-3
)

best_model_path = "/home/boadem/Work/School/Unet_brain_mri_models_WGM/best_model_fold_4.pth"
state_dict = torch.load(best_model_path, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded and ready for inference.")

test_set = json.load(open("/home/boadem/Work/School/test_set_paths.json"))
from unet_mri_utils import BrainMRIDataset, MergeSegLabels, AddChannelDim
from monai.transforms import Compose, ResizeD, ScaleIntensityRanged, EnsureTyped
from torch.utils.data import DataLoader

labels_file_path = "/home/boadem/Work/School/neurite_data/seg24_labels.txt"

test_transforms = Compose([
    AddChannelDim(keys=["image", "label"]),
    MergeSegLabels(keys=["label"], labels_file_path=labels_file_path),
    ResizeD(keys=["image", "label"], spatial_size=(128,128), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
])

test_dataset = BrainMRIDataset(
            img_paths=[item["image"] for item in test_set],
            seg_paths=[item["segmentation"] for item in test_set],
            transform=test_transforms
        )

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
dice_scores = []
dice_metric = DiceMetric(include_background=False, reduction="mean")

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)

        outputs = model(images)

        outputs_onehot = F.one_hot(torch.argmax(outputs, dim=1), num_classes=3).permute(0,3,1,2).float()
        segs_onehot = F.one_hot(labels.squeeze(1), num_classes=3).permute(0,3,1,2).float()

        dice_metric(y_pred=outputs_onehot, y=segs_onehot)
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)

        print(f"Batch {batch_idx+1}, Dice Score: {dice_score:.4f}")

dice_metric.reset()

average_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)
print(f"Average Dice Score over Test Set: {average_dice:.4f} ± {std_dice:.4f}")

with open("/home/boadem/Work/School/unet_mri_test_results.txt", "w") as f:
    f.write(f"Average Dice Score: {average_dice:.4f} ± {std_dice:.4f}\n")