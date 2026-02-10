import voxelmorph_mri_utils as utils
from sklearn.model_selection import KFold
import os
import pytorch_lightning as pl
import torch
import json



def train(data, save_dir, log_save_dir, num_classes, n_epochs=60, patience=10, batch_size=4):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42) #Splitting the data into 5 folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\nStarting fold {fold+1}...")

        train_paths = [data[i] for i in train_idx]
        val_paths = [data[i] for i in val_idx]

        train_dataset = utils.BrainMRIDataset(
            image_paths=[item["image"] for item in train_paths],
            seg_paths=[item["segmentation"] for item in train_paths],
            partition="train",
            num_classes=num_classes
        )
        
        val_dataset = utils.BrainMRIDataset(
            image_paths=[item["image"] for item in val_paths],
            seg_paths=[item["segmentation"] for item in val_paths],
            partition="val",
            num_classes=num_classes
        )

        data_module = utils.BrainMRIDataModule(train_ds=train_dataset,
                                               val_ds=val_dataset,
                                               batch_size=batch_size)

        model = utils.VoxelMorphModel(image_shape=(128, 128), num_classes=num_classes, alpha=0.01, beta=0.2, lr=5e-4)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_SSIM", #We checkpoint at the best validation SSIM
            dirpath=save_dir,
            filename=f"best_vxm_fold_{fold+1}",
            mode="max"
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_SSIM", #We monitor the validation SSIM for early stopping
            patience=patience, 
            mode="max")
        logger = pl.loggers.TensorBoardLogger(log_save_dir, name=f"vxm_fold_{fold+1}") #Logging training and validation processes

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            log_every_n_steps=10,
            accelerator="gpu",
            devices=1
        )

        trainer.fit(model, data_module)

        model_path = os.path.join(save_dir, f"best_model_fold_{fold+1}.pth")
        trainer.save_checkpoint(os.path.join(save_dir, f"checkpoint_fold_{fold+1}.ckpt"))
        torch.save(model.model.state_dict(), os.path.join(save_dir, f"weights_fold_{fold+1}.pth"))
        
        print(f"Best model for fold {fold+1} saved at: {model_path}")

    print("Training complete for all folds.")


train_set = json.load(open("train_set_paths.json"))

num_classes = 3
train(
    data=train_set,
    save_dir=f"/VoxelMorph_/trained_models/Vxm_brain_mri_models_V2_{num_classes}labels",
    log_save_dir=f"/VoxelMorph_/logs/Vxm_brain_mri_logs_V2_{num_classes}labels",
    num_classes=num_classes,
    n_epochs=100,
    patience=10
)
