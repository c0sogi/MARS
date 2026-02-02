import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_metrics
from library.losses import BCEDiceLoss
from library.model import UNet25D
from library.dataset import UWGIDataset, get_transforms


def load_data(load_cached_data=True):
    """
    Loads training and validation metadata.

    Args:
        load_cached_data (bool): Flag to indicate if cached data should be used.
                                 (Here we load directly from the pre-generated metadata CSVs
                                  as they serve as the cache).
    Returns:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
    """
    # The metadata CSVs are already the result of the processing step
    # so we load them directly.
    if os.path.exists(Config.TRAIN_CSV) and os.path.exists(Config.VAL_CSV):
        train_df = pd.read_csv(Config.TRAIN_CSV, keep_default_na=False)
        val_df = pd.read_csv(Config.VAL_CSV, keep_default_na=False)
        return train_df, val_df
    else:
        raise FileNotFoundError(
            "Metadata CSV files not found. Please ensure metadata generation is complete."
        )


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks, _ in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Runs validation using 3D reconstruction strategy.
    Predictions are aggregated by case and day to form volumes before metric calculation.
    """
    model.eval()

    # Dictionaries to aggregate slices: key=(case, day), value=list of (slice_num, pred, gt)
    # We use dictionaries to handle the variable number of slices per case
    case_data = {}

    with torch.no_grad():
        for images, masks, ids in loader:
            images = images.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Convert to binary predictions
            preds = (probs > 0.5).float().cpu().numpy()
            gts = masks.cpu().numpy()

            # Parse IDs and aggregate
            for i, slice_id in enumerate(ids):
                # ID format: caseXXX_dayYY_slice_ZZZZ
                parts = slice_id.split("_")
                case_str = parts[0]
                day_str = parts[1]
                slice_num = int(parts[3])

                key = (case_str, day_str)
                if key not in case_data:
                    case_data[key] = []

                # Store (slice_index, prediction_mask, ground_truth_mask)
                # Masks are (C, H, W)
                case_data[key].append((slice_num, preds[i], gts[i]))

    # Compute metrics per case volume
    total_score = 0.0
    total_dice = 0.0
    total_hd = 0.0
    num_cases = len(case_data)

    if num_cases == 0:
        return 0.0, 0.0, 0.0

    for key, slices in case_data.items():
        # Sort slices by index to ensure correct 3D volume construction
        slices.sort(key=lambda x: x[0])

        # Stack to create volumes: Shape (Num_Slices, C, H, W)
        vol_pred_stack = np.stack([s[1] for s in slices], axis=0)
        vol_gt_stack = np.stack([s[2] for s in slices], axis=0)

        # Transpose to (C, Num_Slices, H, W) -> (C, Depth, Height, Width)
        vol_pred_stack = vol_pred_stack.transpose(1, 0, 2, 3)
        vol_gt_stack = vol_gt_stack.transpose(1, 0, 2, 3)

        case_score = 0.0
        case_dice = 0.0
        case_hd = 0.0

        # Calculate metrics for each class channel
        for c in range(Config.NUM_CLASSES):
            # Extract single class 3D volume: (Depth, Height, Width)
            p_vol = vol_pred_stack[c]
            g_vol = vol_gt_stack[c]

            # Calculate metrics
            metrics = compute_metrics(p_vol, g_vol)

            case_score += metrics["score"]
            case_dice += metrics["dice"]
            case_hd += metrics["hausdorff"]

        # Average over classes
        total_score += case_score / Config.NUM_CLASSES
        total_dice += case_dice / Config.NUM_CLASSES
        total_hd += case_hd / Config.NUM_CLASSES

    # Average over cases
    avg_score = total_score / num_cases
    avg_dice = total_dice / num_cases
    avg_hd = total_hd / num_cases

    return avg_score, avg_dice, avg_hd


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False, patience=5
):
    """
    Main execution function for training the model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        debug (bool): If True, runs on a small subset of data.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    train_df, val_df = load_data()

    if debug:
        print("Debug mode: utilizing small subset of data.")
        train_df = train_df.iloc[:200]
        val_df = val_df.iloc[:100]
        epochs = 2

    # 3. Datasets and Loaders
    train_dataset = UWGIDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = UWGIDataset(val_df, transforms=get_transforms("val"), mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Model, Optimizer, Loss
    model = UNet25D(
        backbone_name=Config.BACKBONE, classes=Config.NUM_CLASSES, pretrained=True
    )
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # 5. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    early_stop_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score, val_dice, val_hd = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{epochs} [{elapsed:.0f}s] "
            f"Train Loss: {train_loss} "
            f"Val Score: {val_score} "
            f"Val Dice: {val_dice} "
            f"Val HD: {val_hd}"
        )

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with score {best_score}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Training finished. Best Validation Score: {best_score}")
