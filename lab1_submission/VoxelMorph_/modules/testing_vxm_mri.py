import voxelmorph_mri_utils as utils
from sklearn.model_selection import KFold
import os
import pytorch_lightning as pl
import torch
import json
from layers import SpatialTransformer
from torch.utils.data import DataLoader
import torch. nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from monai.metrics import DiceMetric
from losses_v2 import NCCMetric
from skimage.registration import optical_flow_tvl1
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage.transform import warp






device = torch.device("cuda")
test_set = json.load(open("test_set_paths.json"))
num_classes = 3
labels_file_path = "/neurite_data/seg24_labels.txt"
ckpt_path = "/VoxelMorph_/trained_models/Vxm_brain_mri_models_3labels/best_vxm_fold_3.ckpt"

model = utils.VoxelMorphModel.load_from_checkpoint(ckpt_path, image_shape=(128, 128))
model.eval()
model.cuda() if torch.cuda.is_available() else model.cpu()
spatial_transform_seg = SpatialTransformer(size=(128, 128), mode="nearest").cuda()



test_dataset = utils.BrainMRIDataset(image_paths=[item["image"] for item in test_set],
            seg_paths=[item["segmentation"] for item in test_set],
            partition="val",
            num_classes=num_classes,
            return_2=True)

all_samples = [sample for sample in test_dataset]

test_dataset_vxm = [item[0] for item in all_samples]

test_dataset_pre = [item[1] for item in all_samples]


test_loader_pre = DataLoader(test_dataset_pre, batch_size=1, shuffle=False, num_workers=4)
test_loader_vxm = DataLoader(test_dataset_vxm, batch_size=1, shuffle=False, num_workers=4)





#-------------------------Pre registration--------------------------

pre_PSNR_metric = PSNR(data_range=1.0).to(device)
pre_NCC_metric = NCCMetric().to(device)
pre_SSIM_metric = SSIM().to(device) 
pre_dice_metric = DiceMetric(include_background=False, reduction="mean")

pre_ssim_scores = []
pre_psnr_scores = []
pre_ncc_scores = []

for batch_idx, batch in enumerate(test_loader_pre) : 
    fixed, moving = batch['fixed'].to(device), batch['moving'].to(device)
    fixed_seg, moving_seg = batch['fixed_seg'].to(device), batch['moving_seg'].to(device)
    
    fixed_seg = fixed_seg.squeeze(1).long()
    moving_seg = moving_seg.squeeze(1).long()

    fixed_seg_onehot = F.one_hot(fixed_seg, num_classes=num_classes).permute(0, 3, 1, 2).float()
    moving_seg_onehot = F.one_hot(moving_seg, num_classes=num_classes).permute(0, 3, 1, 2).float()

    psnr = pre_PSNR_metric(moving, fixed).item()
    ssim = pre_SSIM_metric(moving, fixed).item()
    ncc = pre_NCC_metric(moving, fixed).item()
    pre_dice_metric(y_pred=moving_seg_onehot, y=fixed_seg_onehot)

    pre_psnr_scores.append(psnr)
    pre_ssim_scores.append(ssim)
    pre_ncc_scores.append(ncc)


pre_dice_buffer = pre_dice_metric.get_buffer()
pre_mean_dice_per_class = torch.mean(pre_dice_buffer, dim=0).cpu().numpy()
pre_std_dice_per_class = torch.std(pre_dice_buffer, dim=0).cpu().numpy()

pre_overall_avg_dice = np.mean(pre_mean_dice_per_class)
pre_overall_std_dice = np.std(pre_dice_buffer.cpu().numpy())

pre_mean_ssim, pre_std_ssim = np.mean(pre_ssim_scores), np.std(pre_ssim_scores)
pre_mean_ncc, pre_std_ncc = np.mean(pre_ncc_scores), np.std(pre_ncc_scores)
pre_mean_psnr, pre_std_psnr = np.mean(pre_psnr_scores), np.std(pre_psnr_scores)

print(f"\nResults for Pre-registration:")
print(f"Overall Average Dice: {pre_overall_avg_dice:.4f} ± {pre_overall_std_dice:.4f}")

for i, (m, s) in enumerate(zip(pre_mean_dice_per_class, pre_std_dice_per_class)):
    print(f"Class {i} Dice: {m:.4f} ± {s:.4f}")

with open("/VoxelMorph_/vxm_mri_test_results.txt", "a") as f:
    f.write(f"\n--- Result Type: Pre-registration ---\n")
    f.write(f"Overall Average Dice: {pre_overall_avg_dice:.4f} ± {pre_overall_std_dice:.4f}\n")
    
    for i, (m, s) in enumerate(zip(pre_mean_dice_per_class, pre_std_dice_per_class)):
        f.write(f"Class {i+1} Dice: {m:.4f} ± {s:.4f}\n")
        
    f.write("---------------------------\n")
    f.write(f"Overall SSIM : {pre_mean_ssim:.4f} ± {pre_std_ssim:.4f}\n")
    f.write(f"Overall PSNR : {pre_mean_psnr:.4f} ± {pre_std_psnr:.4f}\n")
    f.write(f"Overall NCC  : {pre_mean_ncc:.4f} ± {pre_std_ncc:.4f}\n")

pre_dice_metric.reset()
pre_SSIM_metric.reset()
pre_NCC_metric.reset()
pre_PSNR_metric.reset()




#---------------------Optical Flow Estimation/ Phase Correlation -------------------------

of_ssim_scores = []
of_psnr_scores = []
of_ncc_scores = []
of_folding_ratios = []

device = "cpu"

of_PSNR_metric = PSNR(data_range=1.0).to(device)
of_NCC_metric = NCCMetric().to(device)
of_SSIM_metric = SSIM().to(device) 
of_dice_metric = DiceMetric(include_background=False, reduction="mean")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader_vxm) : 
        fixed_tensor = batch['fixed'].to(device)
        moving_tensor = batch['moving'].to(device)
        moving_seg_tensor = batch['moving_seg'].to(device)
        
        fixed_np = fixed_tensor.detach().cpu().numpy().squeeze()
        moving_np = moving_tensor.detach().cpu().numpy().squeeze()
        moving_seg_np = moving_seg_tensor.detach().cpu().numpy().squeeze()

        # v, u = optical_flow_tvl1(reference_image=fixed_np, moving_image=moving_np)  
        # nr, nc = fixed_np.shape
        # row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

        # moved_np = warp(moving_np, np.array([row_coords + v, col_coords + u]), mode='edge')
        # warped_seg_np = warp(moving_seg_np, np.array([row_coords + v, col_coords + u]), order=0, mode='edge', preserve_range=True)

        shift_vector, error, diff_phase  = phase_cross_correlation(fixed_np, moving_np)
        moved_np = shift(moving_np, shift_vector)
        warped_seg_np = shift(moving_seg_np, shift_vector, mode="nearest", order=0)

        moved_t = torch.from_numpy(moved_np).float().to(device).unsqueeze(0).unsqueeze(0)
        fixed_t = fixed_tensor 
        warped_seg_t = torch.from_numpy(warped_seg_np).long().to(device).unsqueeze(0).unsqueeze(0)
        fixed_seg_t = batch['fixed_seg'].to(device).long()

        psnr = of_PSNR_metric(moved_t, fixed_t).item()
        ssim = of_SSIM_metric(moved_t, fixed_t).item()
        ncc = of_NCC_metric(moved_t, fixed_t).item()
        
        f_seg_oh = F.one_hot(fixed_seg_t.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        w_seg_oh = F.one_hot(warped_seg_t.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        of_dice_metric(y_pred=w_seg_oh, y=f_seg_oh)

        # flow_of = torch.from_numpy(np.stack([u, v], axis=0)).float().to(device).unsqueeze(0)
        # jac = model.jacobian_determinant(flow_of)
        # folding_ratio = (jac <= 0).float().mean().item()

        of_psnr_scores.append(psnr)
        of_ssim_scores.append(ssim)
        of_ncc_scores.append(ncc)
        # of_folding_ratios.append(folding_ratio)

of_dice_buffer = of_dice_metric.get_buffer()
of_mean_dice_per_class = torch.mean(of_dice_buffer, dim=0).cpu().numpy()
of_std_dice_per_class = torch.std(of_dice_buffer, dim=0).cpu().numpy()


of_overall_avg_dice = np.mean(of_mean_dice_per_class)
of_overall_std_dice = np.std(of_dice_buffer.cpu().numpy())

of_mean_ssim, of_std_ssim = np.mean(of_ssim_scores), np.std(of_ssim_scores)
of_mean_ncc, of_std_ncc = np.mean(of_ncc_scores), np.std(of_ncc_scores)
of_mean_psnr, of_std_psnr = np.mean(of_psnr_scores), np.std(of_psnr_scores)
# of_mean_fr, of_std_fr = np.mean(of_folding_ratios), np.std(of_folding_ratios)

print(f"\nResults for Phase Cross Correlation:")

print(f"Overall Average Dice: {of_overall_avg_dice:.4f} ± {of_overall_std_dice:.4f}")

for i, (m, s) in enumerate(zip(of_mean_dice_per_class, of_std_dice_per_class)):
    print(f"Class {i} Dice: {m:.4f} ± {s:.4f}")

with open("/VoxelMorph_/vxm_mri_test_results.txt", "a") as f:
    # f.write(f"\n--- Result Type: Optical Flow Estimation ---\n")
    f.write(f"\n--- Result Type: Phase Cross Correlation ---\n")

    f.write(f"Overall Average Dice: {of_overall_avg_dice:.4f} ± {of_overall_std_dice:.4f}\n")
    
    for i, (m, s) in enumerate(zip(of_mean_dice_per_class, of_std_dice_per_class)):
        f.write(f"Class {i+1} Dice: {m:.4f} ± {s:.4f}\n")
        
    f.write("---------------------------\n")
    f.write(f"Overall SSIM : {of_mean_ssim:.4f} ± {of_std_ssim:.4f}\n")
    f.write(f"Overall PSNR : {of_mean_psnr:.4f} ± {of_std_psnr:.4f}\n")
    f.write(f"Overall NCC  : {of_mean_ncc:.4f} ± {of_std_ncc:.4f}\n")
    # f.write(f"Overall folding R : {of_mean_fr:.4f} ± {of_std_fr:.4f}\n")

of_dice_metric.reset()
of_SSIM_metric.reset()
of_NCC_metric.reset()
of_PSNR_metric.reset()



#-------------------------------VoxelMorph----------------------------

test_datasets = utils.BrainMRIDataset(image_paths=[item["image"] for item in test_set],
            seg_paths=[item["segmentation"] for item in test_set],
            partition="val",
            num_classes=num_classes,
            return_2=False)


test_loader_vxm = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)
device = torch.device("cuda")

ssim_scores = []
psnr_scores = []
ncc_scores = []
folding_ratios = []

PSNR_metric = PSNR(data_range=1.0).to(device)
NCC_metric = NCCMetric().to(device)
SSIM_metric = SSIM().to(device) 
dice_metric = DiceMetric(include_background=False, reduction="mean")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader_vxm) : 
        fixed, moving = batch['fixed'].to(device), batch['moving'].to(device)
        fixed_seg, moving_seg = batch['fixed_seg'].to(device), batch['moving_seg'].to(device)
        
        moved, flow = model.model(moving, fixed, registration=True)
        warped_seg = spatial_transform_seg(moving_seg, flow)
        

        # Shape: (B, H, W)
        fixed_seg = fixed_seg.squeeze(1).long()
        warped_seg = warped_seg.squeeze(1).long()

        fixed_seg_onehot = F.one_hot(fixed_seg, num_classes=num_classes).permute(0, 3, 1, 2).float()
        warped_seg_onehot = F.one_hot(warped_seg, num_classes=num_classes).permute(0, 3, 1, 2).float()

        psnr = PSNR_metric(moved, fixed).item()
        ssim = SSIM_metric(moved, fixed).item()
        ncc = NCC_metric(moved, fixed).item()
        dice_metric(y_pred=warped_seg_onehot, y=fixed_seg_onehot)

        jac = model.jacobian_determinant(flow)
        folding_ratio = (jac <= 0).cpu().float().mean() 

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        ncc_scores.append(ncc)
        folding_ratios.append(folding_ratio)

dice_buffer = dice_metric.get_buffer()
mean_dice_per_class = torch.mean(dice_buffer, dim=0).cpu().numpy()
std_dice_per_class = torch.std(dice_buffer, dim=0).cpu().numpy()

print(mean_dice_per_class)

overall_avg_dice = np.mean(mean_dice_per_class)
overall_std_dice = np.std(dice_buffer.cpu().numpy())

mean_ssim, std_ssim = np.mean(ssim_scores), np.std(ssim_scores)
mean_ncc, std_ncc = np.mean(ncc_scores), np.std(ncc_scores)
mean_psnr, std_psnr = np.mean(psnr_scores), np.std(psnr_scores)
mean_fr, std_fr = np.mean(folding_ratios), np.std(folding_ratios)

print(f"\nResults for VoxelMorph:")
print(f"Overall Average Dice: {overall_avg_dice:.4f} ± {overall_std_dice:.4f}")

for i, (m, s) in enumerate(zip(mean_dice_per_class, std_dice_per_class)):
    print(f"Class {i} Dice: {m:.4f} ± {s:.4f}")

with open("/VoxelMorph_/vxm_mri_test_results.txt", "a") as f:
    f.write(f"\n--- Result Type: VoxelMorph ---\n")
    f.write(f"Overall Average Dice: {overall_avg_dice:.4f} ± {overall_std_dice:.4f}\n")
    
    for i, (m, s) in enumerate(zip(mean_dice_per_class, std_dice_per_class)):
        f.write(f"Class {i+1} Dice: {m:.4f} ± {s:.4f}\n")
        
    f.write("---------------------------\n")
    f.write(f"Overall SSIM : {mean_ssim:.4f} ± {std_ssim:.4f}\n")
    f.write(f"Overall PSNR : {mean_psnr:.4f} ± {std_psnr:.4f}\n")
    f.write(f"Overall NCC  : {mean_ncc:.4f} ± {std_ncc:.4f}\n")
    f.write(f"Overall folding R : {mean_fr:.4f} ± {std_fr:.4f}\n")

dice_metric.reset()
SSIM_metric.reset()
NCC_metric.reset()
PSNR_metric.reset()



