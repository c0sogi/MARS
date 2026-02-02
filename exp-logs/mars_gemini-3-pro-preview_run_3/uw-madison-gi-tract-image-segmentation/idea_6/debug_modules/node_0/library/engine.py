import os
import gc
import numpy as np
import torch
import cv2
import pandas as pd
from collections import defaultdict
from library.config import Config
from library.utils import (
    compute_dice_coefficient,
    compute_hausdorff_3d,
    apply_3d_cca,
    rle_decode,
)


def train_one_epoch(
    model, loader, optimizer, scheduler, loss_fn, scaler, epoch, device
):
    """
    Runs one epoch of training with Deep Supervision and Curriculum Loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            # Model output can be a list (Deep Supervision) or tensor
            outputs = model(images)
            loss = loss_fn(outputs, masks, epoch)

        scaler.scale(loss).backward()
        scaler.scale(optimizer).step()
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step scheduler if it's per-iteration (not used here, but good practice)
    # Note: ReduceLROnPlateau is stepped after validation in the fit function

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def inverse_transform_slice(pred_mask, padding_info):
    """
    Reverses the resize_pad_to_square operation for a single slice.

    Args:
        pred_mask (np.ndarray): (C, H_padded, W_padded) probability map.
        padding_info (dict): Padding and scaling metadata.

    Returns:
        np.ndarray: (C, H_orig, W_orig) restored probability map.
    """
    h_orig = int(padding_info["original_height"])
    w_orig = int(padding_info["original_width"])
    h_new = int(padding_info["new_height"])
    w_new = int(padding_info["new_width"])
    pad_top = int(padding_info["pad_top"])
    pad_left = int(padding_info["pad_left"])

    # 1. Crop padding
    # pred_mask is (C, H, W)
    cropped = pred_mask[:, pad_top : pad_top + h_new, pad_left : pad_left + w_new]

    # 2. Resize back to original dimensions
    # Transpose to (H, W, C) for cv2
    cropped_tr = np.transpose(cropped, (1, 2, 0))

    # Linear interpolation for probabilities
    resized = cv2.resize(cropped_tr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Handle single channel case (cv2 removes last dim)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    # Transpose back to (C, H, W)
    restored = np.transpose(resized, (2, 0, 1))

    return restored


def validate_3d(model, loader, device):
    """
    Performs validation by reconstructing 3D volumes in the original coordinate space.
    """
    model.eval()

    # Storage: preds_map[case_day][slice_num] = pred_numpy_c_h_w
    preds_map = defaultdict(dict)

    # Access the dataframe from the dataset to get Ground Truth RLEs
    val_df = loader.dataset.df

    # --- 1. Inference Phase ---
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["id"]

            # Padding info is a dict of lists/tensors
            pad_info_batch = batch["padding_info"]

            # Forward pass
            outputs = model(images)

            # Handle Deep Supervision: take the final (highest resolution) output
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            preds = torch.sigmoid(outputs).cpu().numpy()

            # Process each sample in the batch
            for i, sample_id in enumerate(ids):
                # Parse ID: caseXXX_dayYY_slice_ZZZZ
                parts = sample_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                # Extract padding info for this specific sample
                p_info = {
                    k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i]
                    for k, v in pad_info_batch.items()
                }

                # Inverse Transform to original geometry
                restored_pred = inverse_transform_slice(preds[i], p_info)

                preds_map[case_day][slice_num] = restored_pred

    # --- 2. Metric Calculation Phase ---
    dice_scores = []
    hausdorff_scores = []

    # Group validation dataframe by case_day to reconstruct volumes
    groups = val_df.groupby(["case", "day"])

    for (case, day), group in groups:
        case_day = f"{case}_{day}"

        if case_day not in preds_map:
            continue

        # Get sorted slice indices
        slice_indices = sorted(group["slice"].astype(int).tolist())

        # Dimensions
        h_orig = group.iloc[0]["height"]
        w_orig = group.iloc[0]["width"]
        depth = len(slice_indices)
        num_classes = Config.NUM_CLASSES

        # Initialize 3D Volumes: (Depth, Height, Width, Classes)
        gt_vol = np.zeros((depth, h_orig, w_orig, num_classes), dtype=np.uint8)
        pred_vol = np.zeros((depth, h_orig, w_orig, num_classes), dtype=np.float32)

        # Map slice number to row for fast RLE lookup
        slice_to_row = {row["slice"]: row for _, row in group.iterrows()}

        for z, s_idx in enumerate(slice_indices):
            # Fill Ground Truth
            row = slice_to_row[s_idx]
            for c_idx, cls_name in enumerate(Config.CLASSES):
                rle = row[cls_name]
                if pd.notna(rle) and rle != "":
                    gt_vol[z, :, :, c_idx] = rle_decode(rle, (h_orig, w_orig))

            # Fill Prediction
            if s_idx in preds_map[case_day]:
                # stored pred is (C, H, W), transpose to (H, W, C)
                p_slice = np.transpose(preds_map[case_day][s_idx], (1, 2, 0))
                pred_vol[z, :, :, :] = p_slice

        # Binarize Predictions
        pred_vol_bin = (pred_vol > Config.MASK_THRESHOLD).astype(np.uint8)

        # Compute Metrics per Class
        case_dices = []
        case_hds = []

        for c_idx in range(num_classes):
            g_c = gt_vol[:, :, :, c_idx]  # (D, H, W)
            p_c = pred_vol_bin[:, :, :, c_idx]  # (D, H, W)

            # Apply 3D Connected Component Analysis
            p_c_processed = apply_3d_cca(p_c)

            # Dice
            d = compute_dice_coefficient(g_c, p_c_processed)
            case_dices.append(d)

            # Hausdorff Distance (Normalized 0-1)
            h = compute_hausdorff_3d(g_c, p_c_processed)
            case_hds.append(h)

        dice_scores.append(np.mean(case_dices))
        hausdorff_scores.append(np.mean(case_hds))

    # Aggregate
    mean_dice = np.mean(dice_scores)
    mean_hd = np.mean(hausdorff_scores)

    # Competition Score: 0.4 * Dice + 0.6 * (1 - HD)
    # Note: HD is a distance (0 is best), so we invert it for the score (1 is best)
    score = (Config.DICE_WEIGHT * mean_dice) + (
        Config.HAUSDORFF_WEIGHT * (1.0 - mean_hd)
    )

    metrics = {"val_dice": mean_dice, "val_hd": mean_hd, "val_score": score}

    return metrics


def fit(model, train_loader, val_loader, optimizer, scheduler, loss_fn, config):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    device = config.DEVICE
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_score = -np.inf
    patience_counter = 0
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # --- Training ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler, epoch, device
        )

        # --- Validation ---
        val_metrics = validate_3d(model, val_loader, device)

        # --- Logging ---
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Dice  : {val_metrics['val_dice']:.6f}")
        print(f"  Val HD    : {val_metrics['val_hd']:.6f}")
        print(f"  Val Score : {val_metrics['val_score']:.6f}")

        # --- Scheduler Step ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_score"])
            else:
                scheduler.step()

        # --- Checkpointing & Early Stopping ---
        current_score = val_metrics["val_score"]

        # Save Last Model
        last_path = os.path.join(config.CHECKPOINT_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_path)

        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  New Best Score! Model saved to {best_path}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best Score: {best_score:.6f}")
