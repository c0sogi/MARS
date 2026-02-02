import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_rmse,
    save_checkpoint,
    load_checkpoint,
)
from library.model import ResUNetPlusPlus
from library.dataset import DenoisingDataset


# -------------------------------------------------------------------------
# Loss Function
# -------------------------------------------------------------------------
class DeepSupervisionLoss(nn.Module):
    """
    Computes the weighted sum of MSE losses for Deep Supervision.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, target):
        # If deep supervision is enabled, preds is a list of tensors
        if isinstance(preds, list):
            loss = 0.0
            for pred in preds:
                loss += self.mse(pred, target)
            return loss / len(preds)
        else:
            # Fallback if model returns single tensor
            return self.mse(preds, target)


# -------------------------------------------------------------------------
# Helper: Padding for Inference
# -------------------------------------------------------------------------
def pad_to_multiple(x, multiple=16):
    """
    Pads input tensor (B, C, H, W) so H and W are multiples of 'multiple'.
    Returns padded tensor and original dimensions (h, w).
    """
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # Pad format: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x, h, w


def unpad(x, h, w):
    """
    Crops tensor to original dimensions h, w.
    """
    return x[:, :, :h, :w]


# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for noisy, target in loader:
        noisy = noisy.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Returns list of outputs if deep supervision is on
        outputs = model(noisy)

        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    return running_loss / len(loader.dataset)


# -------------------------------------------------------------------------
# Validation Loop
# -------------------------------------------------------------------------
def validate(model, loader, device):
    model.eval()

    sse = 0.0  # Sum of squared errors
    total_pixels = 0

    with torch.no_grad():
        for noisy, target in loader:
            noisy = noisy.to(device)
            target = target.to(device)  # This is the noise residual

            # Pad input for network compatibility
            noisy_padded, h, w = pad_to_multiple(noisy, 16)

            # Inference (returns single tensor in eval mode)
            pred_residual_padded = model(noisy_padded)

            # Unpad
            pred_residual = unpad(pred_residual_padded, h, w)

            # Reconstruct Clean Images
            # Ground Truth Clean = Noisy - Target
            gt_clean = torch.clamp(noisy - target, 0, 1)

            # Predicted Clean = Noisy - Pred_Residual
            pred_clean = torch.clamp(noisy - pred_residual, 0, 1)

            # Accumulate Error
            diff = (pred_clean - gt_clean).cpu().numpy().flatten()
            sse += np.sum(diff**2)
            total_pixels += diff.size

    rmse = np.sqrt(sse / total_pixels)
    return rmse


# -------------------------------------------------------------------------
# Submission Generation
# -------------------------------------------------------------------------
def generate_submission(model, device, output_path=Config.SUBMISSION_FILE_PATH):
    print("Generating submission...")

    # Load Test Dataset
    test_dataset = DenoisingDataset(mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    model.eval()
    results = []

    with torch.no_grad():
        for noisy, img_id in test_loader:
            noisy = noisy.to(device)
            img_id = img_id[0]  # Extract string from tuple

            # Pad
            noisy_padded, h, w = pad_to_multiple(noisy, 16)

            # TTA: Test Time Augmentation
            # 1. Original
            pred_res_1 = model(noisy_padded)

            # 2. Horizontal Flip
            noisy_h = torch.flip(noisy_padded, [3])
            pred_res_h = model(noisy_h)
            pred_res_2 = torch.flip(pred_res_h, [3])

            # 3. Vertical Flip
            noisy_v = torch.flip(noisy_padded, [2])
            pred_res_v = model(noisy_v)
            pred_res_3 = torch.flip(pred_res_v, [2])

            # Average predictions
            avg_residual = (pred_res_1 + pred_res_2 + pred_res_3) / 3.0

            # Unpad
            pred_residual = unpad(avg_residual, h, w)

            # Reconstruct Clean
            # Note: noisy is the original unpadded tensor on device
            pred_clean = torch.clamp(noisy - pred_residual, 0, 1)

            # Convert to numpy
            pred_clean_np = pred_clean.squeeze().cpu().numpy()  # (H, W)

            # Melt to pixels
            rows, cols = pred_clean_np.shape
            for r in range(rows):
                for c in range(cols):
                    # 1-based indexing for row/col
                    pixel_id = f"{img_id}_{r+1}_{c+1}"
                    val = pred_clean_np[r, c]
                    results.append({"id": pixel_id, "value": val})

    # Create DataFrame and Save
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# -------------------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------------------
def run_training(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
    seed_everything()
    device = Config.DEVICE
    print(f"Starting training on {device}...")

    # Datasets
    # Note: Dataset handles caching internally
    train_dataset = DenoisingDataset(mode="train")
    val_dataset = DenoisingDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full image validation requires batch_size=1
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = ResUNetPlusPlus().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.ETA_MIN
    )

    criterion = DeepSupervisionLoss()

    # Training State
    best_rmse = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Print Metrics (Full precision for Val RMSE)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Time: {elapsed:.1f}s | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse}"
        )

        # Checkpointing & Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best RMSE: {best_rmse}"
            )
            break

    print(f"Training complete. Best RMSE: {best_rmse}")

    # Generate Submission with Best Model
    # Reload best weights
    load_checkpoint(model, Config.MODEL_SAVE_PATH, device=device)
    generate_submission(model, device)
