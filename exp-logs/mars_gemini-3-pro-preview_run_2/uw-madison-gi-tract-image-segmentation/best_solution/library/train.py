import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, dice_coef, hausdorff_3d
from library.ghost_model import GhostUNet, bce_dice_loss
from library.dataset import process_metadata, get_dataloaders


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = bce_dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Validates the model by reconstructing 3D volumes and calculating
    Dice and 3D Hausdorff distance.
    """
    model.eval()

    # Storage for reconstructing 3D volumes
    # Structure: data_store[case_day][slice_id] = {'pred': np.array, 'true': np.array}
    data_store = defaultdict(dict)

    with torch.no_grad():
        for images, masks, ids in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Move to CPU
            preds = preds.cpu().numpy()
            masks = masks.numpy()

            # Group by case
            for i, img_id in enumerate(ids):
                # ID format: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                data_store[case_day][slice_num] = {
                    "pred": preds[i],  # Shape (C, H, W)
                    "true": masks[i],  # Shape (C, H, W)
                }

    # Calculate metrics over 3D volumes
    dice_scores = []
    hausdorff_scores = []

    for case_day, slices_data in data_store.items():
        # Sort slices by index to form a proper volume
        sorted_slice_indices = sorted(slices_data.keys())

        # Stack slices to form (Depth, C, H, W)
        vol_pred = np.stack([slices_data[s]["pred"] for s in sorted_slice_indices])
        vol_true = np.stack([slices_data[s]["true"] for s in sorted_slice_indices])

        # Transpose to (C, Depth, H, W) for iterating over classes
        vol_pred = vol_pred.transpose(1, 0, 2, 3)
        vol_true = vol_true.transpose(1, 0, 2, 3)

        # Calculate metrics per class
        for c in range(Config.NUM_CLASSES):
            y_p = vol_pred[c]
            y_t = vol_true[c]

            # Dice
            d = dice_coef(y_t, y_p)
            dice_scores.append(d)

            # Hausdorff 3D
            # Note: hausdorff_3d handles empty masks internally
            h = hausdorff_3d(y_t, y_p)

            # Handle infinity (if one is empty and other is not, dist is inf)
            # We cap it or treat it as max distance (1.0) since coordinates are normalized
            if np.isinf(h):
                h = 1.0

            hausdorff_scores.append(h)

    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_hausdorff = (
        np.mean(hausdorff_scores) if hausdorff_scores else 1.0
    )  # Default bad score

    # Invert Hausdorff for scoring (lower is better, so we want to maximize 1 - H)
    # The metric definition is 0.4 * Dice + 0.6 * Hausdorff.
    # Usually we want to MINIMIZE Hausdorff.
    # To combine them into a single "Score" to MAXIMIZE, we can use (1 - Hausdorff).
    # However, let's report raw values and use a combined score for checkpointing.
    # Score = 0.4 * Dice + 0.6 * (1 - Hausdorff)

    combined_score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hausdorff)

    return mean_dice, mean_hausdorff, combined_score


def run_training():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    # Using process_metadata from library.dataset
    train_df = process_metadata(
        Config.TRAIN_METADATA_PATH, "train_processed", load_cached_data=True
    )
    val_df = process_metadata(
        Config.VAL_METADATA_PATH, "val_processed", load_cached_data=True
    )

    # Debug mode subsampling
    if Config.DEBUG:
        print("Debug mode: Subsampling data...")
        train_df = train_df.sample(
            frac=Config.DATA_FRACTION, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            frac=Config.DATA_FRACTION, random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Dataloaders
    # Using get_dataloaders from library.dataset which handles balanced sampling
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Setup
    model = GhostUNet(
        in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -np.inf
    patience = 5
    patience_counter = 0

    print("Starting training...")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_dice, val_hausdorff, val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Dice: {val_dice:.6f} | "
            f"Val Hausdorff: {val_hausdorff:.6f} | "
            f"Combined Score: {val_score:.6f}"
        )

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Score: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Combined Score: {best_score:.6f}")
