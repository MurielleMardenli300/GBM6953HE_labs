#!/usr/bin/env python3

# Core library imports
import argparse
from typing import Sequence
from pathlib import Path

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import neurite as ne

# Local imports
import voxelmorph as vxm


class VxmIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for infinite VoxelMorph registration data.
    """

    def __init__(self, device: str = 'cpu', data_dir: str = None):
        self.device = device
        self.oasis_path = Path(data_dir or '.')  # Uses --output
        self._get_vol_paths()


    def __iter__(self):
        while True:
            if len(self.folder_abspaths) == 0:
                print("ERROR: No valid NPZ files!")
                return
            
            idx1, idx2 = np.random.randint(0, len(self.folder_abspaths), 2)
            
            # LOAD NPZ CORRECTLY
            source_data = np.load(self.folder_abspaths[idx1])
            target_data = np.load(self.folder_abspaths[idx2])
            
            source = torch.from_numpy(source_data['vol']).unsqueeze(0).float()
            target = torch.from_numpy(target_data['vol']).unsqueeze(0).float()
            
            yield {'source': source, 'target': target}


    #==== CHANGED FUNCTION ====
    def _get_vol_paths(self):
        self.folder_abspaths = []
        print(f"Scanning: {self.oasis_path}")
        for npz_file in self.oasis_path.glob('*.npz'):
            self.folder_abspaths.append(npz_file)
        print(f"Found {len(self.folder_abspaths)} NPZ files: {self.folder_abspaths[:3]}")




def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    steps_per_epoch: int,
    device: str = 'cuda'
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    steps_per_epoch : int
    """

    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        batch = next(dataloader)
        optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement'
        )

        img_loss = image_loss_fn(target, warped_source)
        grad_loss = grad_loss_fn(displacement)

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('--output', type=str, default='model_3d.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01)
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument(
        '--warm-start', type=int, default=10, help='Early stopping warm start steps'
    )
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')



    #======== CHANGED TO 2D ============= 
    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=2, 
        source_channels=1,
        target_channels=1,
        nb_features=[32, 32, 32],
        integration_steps=0,
    ).to(device)

    # Setup losses and optimizer
    image_loss_fn = ne.nn.modules.MSE()
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmIterableDataset(device=device, data_dir=args.output)
    sample = next(iter(train_dataset))  # Test first
    print(f"Sample shapes: source={sample['source'].shape}, target={sample['target'].shape}")

    train_loader = iter(
        DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    )


    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_loss = float('inf')
    loss_history = []
    for epoch in tqdm(range(args.epochs), desc='Epochs'):

        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=args.steps_per_epoch,
            device=device
        )

        # Track loss history
        loss_history.append(avg_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}')

        # Early stopping check
        if ne.utils.early_stopping(
            loss_history,
            patience=args.patience,
            threshold=args.threshold,
            warm_start_steps=args.warm_start
        ):
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:

            # Build checkpoint file name
            checkpoint_path = output_path.parent.joinpath(
                f'{output_path.stem}_default-int_epoch{epoch + 1}.pt'
            )

            # Save
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_path.parent / f'{output_path.stem}_best.pt'
            torch.save(model.state_dict(), best_path)

    # Save final model
    final_path = output_path.parent / f"{output_path.stem}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')



if __name__ == '__main__':
    main()
