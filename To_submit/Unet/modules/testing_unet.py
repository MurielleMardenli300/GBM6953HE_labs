import torch
import json
import torch.nn.functional as F
from Unet.modules.unet_mri_utils import UNetModule  # your module
from monai.metrics import DiceMetric, compute_hausdorff_distance
import numpy as np

from Unet.modules.unet_mri_utils import BrainMRIDataset, MergeSegLabels, AddChannelDim
from monai.transforms import Compose, ResizeD, ScaleIntensityRanged, EnsureTyped
from torch.utils.data import DataLoader

from monai.networks.utils import one_hot
import tqdm
from skimage.filters import threshold_multiotsu

type = "4labels"

if type == "WGM":
    num_classes = 3
    labels_file_path = "/neurite_data/seg24_labels.txt"
    train_set_path = "train_set_paths.json"
    test_set_path = "test_set_paths.json"
elif type == "4labels":
    num_classes = 5
    labels_file_path = "/neurite_data/seg4_labels.txt"
    train_set_path = "train_set_paths_4labels.json"
    test_set_path = "test_set_paths_4labels.json"
    
model = UNetModule(
    learning_rate=1e-4,
    num_classes=num_classes
)

best_model_path = f"/Unet/logs/Unet_brain_mri_models_{type}/best_model_fold_1.pth"

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


def map_otsu_to_gt_by_intensity(img, seg_otsu, seg_gt, mask, num_tissue_classes):
    """
    Map Otsu classes to GT classes by matching mean intensities.
    Assumes both are ordered by intensity.
    """
    gt_mean = {}
    otsu_mean = {}

    for c in range(1, num_tissue_classes + 1):
        gt_mask = (seg_gt == c) & mask
        otsu_mask = (seg_otsu == c) & mask

        gt_mean[c] = img[gt_mask].mean() if gt_mask.sum() > 0 else 0
        otsu_mean[c] = img[otsu_mask].mean() if otsu_mask.sum() > 0 else 0

    gt_sorted = sorted(gt_mean.items(), key=lambda x: x[1])
    otsu_sorted = sorted(otsu_mean.items(), key=lambda x: x[1])

    return {o_cls: g_cls for (o_cls, _), (g_cls, _) in zip(otsu_sorted, gt_sorted)}

def test_multi_otsu(loader, num_classes):
    """
    Multi-Otsu baseline evaluation.

    Args:
        loader: PyTorch DataLoader yielding dicts with keys:
                - "image": (1, H, W, D)
                - "label": (1, H, W, D)
                - optionally "subject_id"
        num_classes: total number of classes INCLUDING background
                     (e.g. 5 = bg + 4 tissues)

    Returns:
        results: dict of metrics per subject
    """
    num_tissue_classes = num_classes - 1

    results = {
        "subject_id": [],
    }
    for c in range(1, num_tissue_classes + 1):
        results[f"dice_class_{c}"] = []
        results[f"precision_class_{c}"] = []
        results[f"recall_class_{c}"] = []
        results[f"hd95_class_{c}"] = []

    dice_metric = DiceMetric(include_background=False, reduction="none")

    for idx, batch in enumerate(loader):
        img = batch["image"].squeeze().cpu().numpy()
        seg = batch["label"].squeeze().cpu().numpy()
        subject_id = batch.get("subject_id", f"sample_{idx}")

        mask = seg > 0
        if mask.sum() == 0:
            continue

        pixels = img[mask]

        # start_time = time.time()

        # Multi-Otsu thresholds
        thresholds = threshold_multiotsu(pixels, classes=num_tissue_classes)

        # Initial Otsu segmentation
        seg_otsu_raw = np.zeros_like(img, dtype=np.int32)
        seg_otsu_raw[mask] = (
            np.digitize(img[mask], bins=thresholds, right=False) + 1
        )

        # Map Otsu → GT classes
        mapping = map_otsu_to_gt_by_intensity(
            img, seg_otsu_raw, seg, mask, num_tissue_classes
        )

        seg_mo = np.zeros_like(img, dtype=np.int32)
        for otsu_c, gt_c in mapping.items():
            seg_mo[seg_otsu_raw == otsu_c] = gt_c

        # inference_time = time.time() - start_time

        # Torch tensors
        preds = torch.from_numpy(seg_mo).long().unsqueeze(0).unsqueeze(0)
        labels = torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)

        preds_oh = torch.nn.functional.one_hot(
            preds, num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        labels_oh = torch.nn.functional.one_hot(
            labels, num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Dice
        dice_metric(y_pred=preds_oh, y=labels_oh)
        dice_per_class = dice_metric.aggregate().cpu().numpy().squeeze()
        dice_metric.reset()

        # Precision / Recall
        # seg_mo_flat = seg_mo[mask].flatten()
        # seg_flat = seg[mask].flatten()

        # precision_per_class = []
        # recall_per_class = []

        # for c in range(1, num_tissue_classes + 1):
        #     pred_bin = (seg_mo_flat == c).astype(int)
        #     true_bin = (seg_flat == c).astype(int)

            # if true_bin.sum() > 0:
            #     precision_per_class.append(
            #         precision_score(true_bin, pred_bin, zero_division=0)
            #     )
            #     recall_per_class.append(
            #         recall_score(true_bin, pred_bin, zero_division=0)
            #     )
            # else:
            #     precision_per_class.append(np.nan)
            #     recall_per_class.append(np.nan)

        # HD95
        hd95_per_class = []
        for c in range(1, num_tissue_classes+1):
            pred_c = (preds_oh[:, c:c+1] > 0.5).float()
            lab_c = (labels_oh[:, c:c+1] > 0.5).float()

            if pred_c.sum() > 0 and lab_c.sum() > 0:
                hd95_per_class.append(
                    compute_hausdorff_distance(pred_c, lab_c, percentile=95).item()
                )
            else:
                hd95_per_class.append(np.nan)

        # Store
        results["subject_id"].append(subject_id)
        # print(len(hd95_per_class))
        for i in range(1, num_tissue_classes+1):
            results[f"dice_class_{i}"].append(dice_per_class[i-1])
            # results[f"precision_class_{i+1}"].append(precision_per_class[i])
            # results[f"recall_class_{i+1}"].append(recall_per_class[i])
            results[f"hd95_class_{i}"].append(hd95_per_class[i-1])

    return results

results = test_multi_otsu(test_loader, num_classes)

# =======================
# Dice aggregation
# =======================
dice_vals = np.stack(
    [results[f"dice_class_{c}"] for c in range(1, num_classes)],
    axis=1
)

dice_avg_per_class = np.nanmean(dice_vals, axis=0)
dice_std_per_class = np.nanstd(dice_vals, axis=0)

dice_avg_total = np.nanmean(np.nanmean(dice_vals, axis=1))
dice_std_total = np.nanstd(np.nanmean(dice_vals, axis=1))

# Console
for i, (m, s) in enumerate(zip(dice_avg_per_class, dice_std_per_class)):
    print(f"  Class {i+1} Dice: {m:.4f} ± {s:.4f}")

# =======================
# HD95 aggregation
# =======================
hd95_vals = np.stack(
    [results[f"hd95_class_{c}"] for c in range(1, num_classes)],
    axis=1
)

hd95_avg_per_class = np.nanmean(hd95_vals, axis=0)
hd95_std_per_class = np.nanstd(hd95_vals, axis=0)

hd95_avg_total = np.nanmean(np.nanmean(hd95_vals, axis=1))
hd95_std_total = np.nanstd(np.nanmean(hd95_vals, axis=1))


with open("/Unet/unet_mri_test_results.txt", "a") as f:
    f.write(f"\n--- Multi-Otsu Results ({type}) ---\n")
    f.write(f"Overall Average Dice: {dice_avg_total:.4f} ± {dice_std_total:.4f}\n")

    for i, (m, s) in enumerate(zip(dice_avg_per_class, dice_std_per_class)):
        f.write(f"Class {i+1} Dice: {m:.4f} ± {s:.4f}\n")

    f.write("\n")
    f.write(f"Overall Average HD95: {hd95_avg_total:.4f} ± {hd95_std_total:.4f}\n")

    for i, (m, s) in enumerate(zip(hd95_avg_per_class, hd95_std_per_class)):
        f.write(f"Class {i+1} HD95: {m:.4f} ± {s:.4f}\n")

    f.write("------------------------------------------\n")


#------------UNET testing--------------------
dice_metric = DiceMetric(include_background=False, reduction="none")

all_hd95 = []  

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1, keepdim=True)

        outputs_onehot = one_hot(preds, num_classes=num_classes)
        segs_onehot = one_hot(labels, num_classes=num_classes)

        # =====================
        # Dice
        # =====================
        dice_metric(y_pred=outputs_onehot, y=segs_onehot)
        batch_dice = dice_metric.get_buffer()[-1]  
        print(batch_dice)

        # =====================
        # HD95
        # =====================
        batch_hd95 = []
        for c in range(1, num_classes):  # skip background
            pred_c = (outputs_onehot[:, c:c+1] > 0.5).float()
            gt_c = (segs_onehot[:, c:c+1] > 0.5).float()

            if pred_c.sum() > 0 and gt_c.sum() > 0:
                hd95 = compute_hausdorff_distance(
                    pred_c, gt_c, percentile=95
                ).item()
            else:
                hd95 = np.nan

            batch_hd95.append(hd95)

        all_hd95.append(batch_hd95)

dice_buffer = dice_metric.get_buffer() 

mean_dice_per_class = torch.mean(dice_buffer, dim=0).cpu().numpy()
std_dice_per_class = torch.std(dice_buffer, dim=0).cpu().numpy()

overall_avg_dice = np.mean(mean_dice_per_class)
overall_std_dice = np.std(dice_buffer.cpu().numpy())


hd95_array = np.array(all_hd95)  

mean_hd95_per_class = np.nanmean(hd95_array, axis=0)
std_hd95_per_class = np.nanstd(hd95_array, axis=0)

overall_avg_hd95 = np.nanmean(mean_hd95_per_class)
overall_std_hd95 = np.nanstd(hd95_array)


print(f"\nResults for {type}:")
print(f"Overall Average Dice: {overall_avg_dice:.4f}")
print(f"Overall Average HD95: {overall_avg_hd95:.4f}")

for i, score in enumerate(mean_dice_per_class):
    print(f"Class {i+1} Dice: {score:.4f}")

for i, score in enumerate(mean_hd95_per_class):
    print(f"Class {i+1} HD95: {score:.4f}")

with open("/Unet/unet_mri_test_results.txt", "a") as f:
    f.write(f"\n--- Unet Results: {type} ---\n")
    f.write(
        f"Overall Average Dice: {overall_avg_dice:.4f} ± {overall_std_dice:.4f}\n"
    )
    f.write(
        f"Overall Average HD95: {overall_avg_hd95:.4f} ± {overall_std_hd95:.4f}\n"
    )

    for i, (m, s) in enumerate(zip(mean_dice_per_class, std_dice_per_class)):
        f.write(f"Class {i+1} Dice: {m:.4f} ± {s:.4f}\n")

    for i, (m, s) in enumerate(zip(mean_hd95_per_class, std_hd95_per_class)):
        f.write(f"Class {i+1} HD95: {m:.4f} ± {s:.4f}\n")

dice_metric.reset()
