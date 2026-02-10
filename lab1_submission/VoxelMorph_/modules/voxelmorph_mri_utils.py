from vxm_networks import VxmDense 
import losses_v2 as losses
from layers import SpatialTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    RandFlipd,
    MapTransform,
    EnsureTyped,
    EnsureChannelFirstd,
    RandGaussianNoised,
    ResizeD
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
import SimpleITK as sitk


nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]



model = VxmDense(inshape=(128, 128), nb_unet_features=nb_features)


class VoxelMorphModel(pl.LightningModule) :

    def __init__(self, image_shape=(128, 128), num_classes = 5, alpha = 0.5, beta = 0.2, lr = 1e-4, nb_features=None) :
        """
        
        :param image_shape: Shape of the input image (2D or 3D)
        :param num_classes: Number of classes for Dice (including the background)
        :param alpha: Contribution of the regularization loss (L2)
        :param beta: Contribution of the dice loss 
        :param lr: Learning rate
        :param nb_features: list of encoder and decoder channels [[encoder], [decoder]]
        """
        
        super().__init__()
        if nb_features is None : 
            nb_features =  [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]

        self.model = VxmDense(inshape=image_shape, nb_unet_features=nb_features)

        self.spatial_transform_seg = SpatialTransformer(size=image_shape, mode="nearest")

        
        #-------Losses-----------------
        # self.criterion_sim = losses.NCC_vxm(win=[9,9])

        # self.criterion_sim = losses.MSE()
        self.criterion_sim = losses.SSIM()

        self.criterion_reg = losses.Grad()
        self.criterion_seg = DiceLoss(to_onehot_y=True, softmax=True)


        self.alpha = alpha 
        self.beta = beta  
        self.lr = lr  

        #-------Metrics-------------
        self.PSNR_metric = PSNR(data_range=1.0)
        self.NCC_metric = losses.NCCMetric()
        self.SSIM_metric = SSIM() 
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.num_classes = num_classes

    def jacobian_determinant(self, flow):
        """
        Computes the Jacobian determinant of a 2D registration flow.
        flow shape: [Batch, 2, H, W]
        """
        # Compute gradients of the flow field
        dy = flow[:, 1:, 1:, :] - flow[:, 1:, :-1, :]
        dx = flow[:, :1, :, 1:] - flow[:, :1, :, :-1]

       
        
        # Extract u and v components
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # Calculate partial derivatives
        du_dx, du_dy = torch.gradient(u, spacing=(1, 1), dim=(-2, -1))
        dv_dx, dv_dy = torch.gradient(v, spacing=(1, 1), dim=(-2, -1))

        det = (du_dx + 1) * (dv_dy + 1) - (du_dy * dv_dx)
        
        return det
        
    def forward(self, x) :
        return self.model(x)

    def training_step(self, batch, batch_idx):
        fixed, moving = batch['fixed'], batch['moving']
        fixed_seg, moving_seg = batch['fixed_seg'], batch['moving_seg']
        
        moved, flow = self.model(moving, fixed, registration=True)
        
        #Encoding to one hot for DiceMetric, each label considered as a channel
        m_seg_oh = F.one_hot(moving_seg.squeeze(1).long(), num_classes=self.num_classes)
        m_seg_oh = m_seg_oh.permute(0, 3, 1, 2).float() 
        warped_seg_oh = self.spatial_transform_seg(m_seg_oh, flow) #warping the segmentations with the generated flow field
        
        #Losses
        loss_sim = self.criterion_sim(fixed, moved)
        # DiceLoss(input=logits/probs, target=labels)
        loss_seg = self.criterion_seg(warped_seg_oh, fixed_seg.long())
        loss_reg = self.criterion_reg(flow) #regularizing the flow field

        total_loss = loss_sim + self.beta * loss_seg + self.alpha * loss_reg
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx) : 
        fixed, moving = batch['fixed'], batch['moving']
        fixed_seg, moving_seg = batch['fixed_seg'], batch['moving_seg']
        
        moved, flow = self.model(moving, fixed, registration=True)
        warped_seg = self.spatial_transform_seg(moving_seg, flow)
        

        fixed_seg = fixed_seg.squeeze(1).long()
        warped_seg = warped_seg.squeeze(1).long()

        #Conversion to one hot for dice computation
        fixed_seg_onehot = F.one_hot(fixed_seg, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        warped_seg_onehot = F.one_hot(warped_seg, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        #Losses computation
        loss_sim = self.criterion_sim(fixed, moved)
        loss_seg = self.criterion_seg(fixed_seg.float().unsqueeze(1), warped_seg.float().unsqueeze(1))
        loss_reg = self.criterion_reg(flow)


        total_loss = loss_sim + self.beta * loss_seg + self.alpha * loss_reg 

        #Metrics computation
        self.PSNR_metric(moved, fixed)
        self.SSIM_metric(moved, fixed)
        self.NCC_metric(moved, fixed)
        self.dice_metric(y_pred=warped_seg_onehot, y=fixed_seg_onehot)

        jac = self.jacobian_determinant(flow)
        folding_ratio = (jac <= 0).float().mean() 
                
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_folding_ratio", folding_ratio, prog_bar=True)

        
        return total_loss
    
    def on_validation_epoch_end(self): 
        self.log("val_PSNR", self.PSNR_metric.compute(), prog_bar=False)
        self.log("val_SSIM", self.SSIM_metric.compute(), prog_bar=True)
        self.log("val_NCC", self.NCC_metric.compute(), prog_bar=False)
        
        dice_score = self.dice_metric.aggregate().item()
        self.log("val_dice", dice_score, prog_bar=True)

        self.PSNR_metric.reset()
        self.SSIM_metric.reset()
        self.NCC_metric.reset()
        self.dice_metric.reset()

    def configure_optimizers(self) : 
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    

np.random.seed(42)
def create_synthetic_moving_data_rigid(fixed_img, fixed_seg, rotation_range=15, shift_range=1):
    """
    Creates a 'Moving' image by applying a random affine transform
    (rotation + translation) to the Fixed image.
    """
    # 1. Generate random parameters
    angle_deg = np.random.uniform(-rotation_range, rotation_range)
    shift_y = np.random.uniform(-shift_range, shift_range)
    shift_x = np.random.uniform(-shift_range, shift_range)

    # print(f"Generating Synthetic Data: Rotation={angle_deg:.2f}°, Shift=({shift_y:.2f}, {shift_x:.2f})")

    # 2. Define the Affine Matrix (Inverse mapping is usually required for scipy)
    # Convert to radians
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix (centered usually requires offset handling,
    # but for simple tasks, direct matrix application is often sufficient
    # if we ignore center-of-rotation artifacts or handle them via 'offset')

    # To rotate around center, we often shift center to origin -> rotate -> shift back.
    # Here we simplify:
    center = np.array(fixed_img.shape) / 2.0
    rotation_mat = np.array([[c, -s], [s, c]])
    offset = center - center.dot(rotation_mat) + np.array([shift_y, shift_x])

    # 3. Apply transformation
    # We use spline interpolation (order=1) for the image
    moving_img = affine_transform(
        fixed_img,
        matrix=rotation_mat,
        offset=offset,
        order=1,
        mode='constant'
    )

    # We use nearest neighbor (order=0) for the segmentation (labels must remain integers)
    moving_seg = affine_transform(
        fixed_seg,
        matrix=rotation_mat,
        offset=offset,
        order=0,
        mode='constant'
    )
    
    return moving_img, moving_seg


def create_synthetic_moving_data_nonrigid(
    fixed_img,
    fixed_seg,
    alpha=15.0,  # at 25 we see visible difference
    sigma=3.5, 
    kx = 0.08,
    ky = 0.1  ):
    """
    Creates a 'Moving' image by applying a smooth non-rigid (elastic)
    deformation to the Fixed image.

    alpha : controls the magnitude of displacement
    sigma : Controls smoothness
    kx, ky : skewing factors along x and y axis respectively
    
    """

    shape = fixed_img.shape

    # Displacement fields
    dx = np.random.randn(*shape)
    dy = np.random.randn(*shape)

    # Smoothen dis fields
    dx = gaussian_filter(dx, sigma=sigma) * alpha
    dy = gaussian_filter(dy, sigma=sigma) * alpha

    # 3. Create meshgrid of coordinates
    x, y = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        indexing='ij'
    )

    # 4. Apply displacement
    coords = np.array([
        x + dx,
        y + dy
    ])

    #Non rigid transformation 1 (skewing along the y axis)

    moving_img, moving_seg = create_synthetic_moving_data_rigid(fixed_img, fixed_seg)


    skewing_matrix_y = np.array(([1, kx],
                             [ky, 1]))
    moving_img = affine_transform(
        moving_img,
        matrix=skewing_matrix_y,
        # offset=,
        order=1,
        mode='constant'
    )

    moving_seg = affine_transform(
        moving_seg,
        matrix=skewing_matrix_y,
        # offset=,
        order=0,
        mode='constant'
    )

    # 5. Warp image (linear interpolation)
    moving_img = map_coordinates(
        moving_img,
        coords,
        order=1,
        mode='constant'
    )

    # 6. Warp segmentation (nearest neighbor)
    moving_seg = map_coordinates(
        moving_seg,
        coords,
        order=0,
        mode='constant'
    )

    # print(
    #     f"Generating Non-Rigid Synthetic Data: "
    #     f"alpha={alpha}, sigma={sigma}"
    # )

    return moving_img, moving_seg


def rigid_register_2d(fixed, moving, moving_seg):

    """ Rigid registration """

    fixed_itk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_itk = sitk.GetImageFromArray(moving.astype(np.float32))
    moving_seg_itk = sitk.GetImageFromArray(moving_seg.astype(np.float32))
    
    for img in [fixed_itk, moving_itk, moving_seg_itk]:
        img.SetOrigin((0, 0))
        img.SetSpacing((1, 1))

    # 2. INITIALIZER
    initial_transform = sitk.Euler2DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_itk, 
        moving_itk, 
        initial_transform, 
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.3)

    # MULTI-RESOLUTION PYRAMID
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 3. OPTIMIZER
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.5, 
        minStep=1e-6, 
        numberOfIterations=200,
        estimateLearningRate=registration.EachIteration
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(initial_transform)
    registration.SetInterpolator(sitk.sitkBSpline) 

    # 4. EXECUTE
    final_transform = registration.Execute(fixed_itk, moving_itk)

    # 5. RESAMPLE
    moving_res = sitk.GetArrayFromImage(
        sitk.Resample(moving_itk, fixed_itk, final_transform, sitk.sitkLinear, 0.0)
    )

    moving_seg_res = sitk.GetArrayFromImage(
        sitk.Resample(
            moving_seg_itk,    
            fixed_itk, 
            final_transform, 
            sitk.sitkNearestNeighbor,
            0.0
        )
    )
    return moving_res, moving_seg_res



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
    
  
class MergeSegLabels(MapTransform):
    def __init__(self, keys, labels_file_path, type="WGM"):
        """ Merging segmentation labels based on the content of the labels _file and the type of segmentation(white, gray matter, or 4 labels)"""
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
  



class BrainMRIDataset(Dataset) : 
    def __init__(self, image_paths, seg_paths, partition, img_size = (128, 128), num_classes=3, return_2=False) : 
        """Return2: If true we return the deformed sample and the rigidly registered sample --> both can be used for inference """
        super().__init__()

        self.img_size = img_size
        self.image_paths  = image_paths
        self.seg_paths = seg_paths
        self.return_2 = return_2 

        if num_classes == 3 : 
            labels_file_path = "/neurite_data/seg24_labels.txt"
        elif num_classes == 5 : 
            labels_file_path = "/neurite_data/seg4_labels.txt"

        if partition == "train" : 
            self.transform = Compose([
                EnsureChannelFirstd(keys=["fixed", "moving", "moving_seg", "fixed_seg"], channel_dim="no_channel"),
                MergeSegLabels(keys=["fixed_seg", "moving_seg"], labels_file_path=labels_file_path, num_classes=num_classes),
                ResizeD(keys=["fixed", "moving", "fixed_seg", "moving_seg"], spatial_size=(128,128), mode=("bilinear", "bilinear", "nearest", "nearest")),
                # ScaleIntensityRanged(keys=["fixed", "moving"], a_min=0, a_max=255, b_min=0.0, b_max=1.0), #Images are already normalized
                RandFlipd(keys=["fixed", "moving", "moving_seg", "fixed_seg"], prob=0.3, spatial_axis=0),
                RandFlipd(keys=["fixed", "moving", "moving_seg", "fixed_seg"], prob=0.3, spatial_axis=1),
                RandGaussianNoised(keys=["fixed", "moving"], prob=0.3, mean=0.0, std=0.1),
                ])
        else : #Validation or Test
            self.transform = Compose([
                EnsureChannelFirstd(keys=["fixed", "moving", "moving_seg", "fixed_seg"], channel_dim="no_channel"),        
                MergeSegLabels(keys=["fixed_seg", "moving_seg"], labels_file_path=labels_file_path, num_classes=num_classes),
                ResizeD(keys=["fixed", "moving", "fixed_seg", "moving_seg"], spatial_size=(128,128), mode=("bilinear", "bilinear", "nearest", "nearest")),
                # ScaleIntensityRanged(keys=["fixed", "moving"], a_min=0, a_max=255, b_min=0.0, b_max=1.0)
            ])

    def __len__(self) : 
        return len(self.image_paths)
    
    def __getitem__(self, idx) : 
        fixed = nib.load(self.image_paths[idx]).get_fdata().squeeze()
        fixed_seg = nib.load(self.seg_paths[idx]).get_fdata().squeeze()

        moving, moving_seg = create_synthetic_moving_data_nonrigid(fixed, fixed_seg) #synthetizing non rigid transformations
        moving_rigid, moving_seg_rigid = rigid_register_2d(fixed=fixed, moving=moving, moving_seg=moving_seg) #rigid registration prior to voxelmorph


        sample_rigid = {"fixed" : fixed.astype(np.float32), "moving" : moving_rigid.astype(np.float32), 
                  "fixed_seg" : fixed_seg.astype(np.int64), "moving_seg" : moving_seg_rigid.astype(np.int64)                     
                  }
        sample_non_rigid = {"fixed" : fixed.astype(np.float32), "moving" : moving_rigid.astype(np.float32), 
                  "fixed_seg" : fixed_seg.astype(np.int64), "moving_seg" : moving_seg_rigid.astype(np.int64)                     
                  }
        
        if self.transform : #transforms
            sample_rigid = self.transform(sample_rigid)
            sample_non_rigid = self.transform(sample_non_rigid)

        if self.return_2 :
            return (sample_non_rigid, sample_rigid)
        else : 
            return sample_rigid
    
class BrainMRIDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size=4, num_workers=4):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
