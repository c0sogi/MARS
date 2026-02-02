import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    set_seed,
    dice_coef,
    hausdorff_distance_3d,
    keep_largest_component_3d,
)
from library.dataset import get_processed_dataframe, balance_dataframe, UWMapDataset
from library.model import UnetPlusPlus
from library.loss import BCETverskyLoss


def train_model(debug=False):
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # 2. Data Preparation
    # Load metadata
    df_train = get_processed_dataframe(Config.TRAIN_METADATA_PATH, split_name="train")
    df_val = get_processed_dataframe(Config.VAL_METADATA_PATH, split_name="val")

    # Balance training data
    df_train = balance_dataframe(df_train, random_state=Config.SEED)

    if debug:
        df_train = df_train.iloc[:100]
        df_val = df_val.iloc[:100]
        epochs = 2
    else:
        epochs = Config.EPOCHS

    # Datasets
    train_dataset = UWMapDataset(df_train, mode="train", img_size=Config.IMG_SIZE)
    val_dataset = UWMapDataset(df_val, mode="val", img_size=Config.IMG_SIZE)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model, Optimizer, Loss
    model = UnetPlusPlus().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)

    criterion = BCETverskyLoss().to(device)

    # 4. Training Loop
    best_score = -1.0
    best_epoch = -1

    print(f"Starting training for {epochs} epochs...")
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
    print(f"Dynamic Scales: {Config.IMG_SCALES}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # --- Train Step ---
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Dynamic Scale Training
            # Randomly resize the batch to one of the configured scales
            target_scale = random.choice(Config.IMG_SCALES)

            if target_scale != Config.IMG_SIZE:
                # Resize images (bilinear)
                images = F.interpolate(
                    images,
                    size=(target_scale, target_scale),
                    mode="bilinear",
                    align_corners=False,
                )
                # Resize masks (nearest to keep binary nature mostly intact, though loss handles floats)
                masks = F.interpolate(
                    masks, size=(target_scale, target_scale), mode="nearest"
                )

            optimizer.zero_grad()

            # Forward pass (returns list of outputs for deep supervision)
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()

        # Dictionaries to aggregate slices by case for 3D metrics
        # Structure: preds[case_id][class_idx] -> list of (slice_idx, mask_2d)
        val_preds_map = {}
        val_gt_map = {}

        # We need to map loader indices back to metadata to reconstruct volumes
        # Since shuffle=False, we can iterate linearly

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)

                # Inference at full resolution (512x512)
                outputs = model(images)  # Returns single tensor in eval mode
                preds = torch.sigmoid(outputs)

                # Convert to numpy
                preds_np = preds.cpu().numpy()
                masks_np = masks.numpy()

                # Map back to case/slice info
                start_idx = batch_idx * Config.BATCH_SIZE
                end_idx = start_idx + images.size(0)
                batch_rows = df_val.iloc[start_idx:end_idx]

                for i, (_, row) in enumerate(batch_rows.iterrows()):
                    case_id = row["case"]
                    slice_idx = row["slice"]

                    if case_id not in val_preds_map:
                        val_preds_map[case_id] = {0: [], 1: [], 2: []}
                        val_gt_map[case_id] = {0: [], 1: [], 2: []}

                    # Store for each class
                    for c in range(Config.NUM_CLASSES):
                        p_slice = (preds_np[i, c] > Config.THRESHOLD).astype(np.uint8)
                        g_slice = masks_np[i, c].astype(np.uint8)

                        val_preds_map[case_id][c].append((slice_idx, p_slice))
                        val_gt_map[case_id][c].append((slice_idx, g_slice))

        # --- Metric Calculation (3D) ---
        dice_scores = []
        hausdorff_scores = []

        for case_id in val_preds_map:
            for c in range(Config.NUM_CLASSES):
                # Retrieve slices and sort by slice index
                p_list = sorted(val_preds_map[case_id][c], key=lambda x: x[0])
                g_list = sorted(val_gt_map[case_id][c], key=lambda x: x[0])

                # Stack to form 3D volume (D, H, W)
                # Note: p_list contains tuples (slice_idx, mask)
                vol_pred = np.stack([x[1] for x in p_list], axis=0)
                vol_gt = np.stack([x[1] for x in g_list], axis=0)

                # Post-processing: Keep largest connected component
                vol_pred = keep_largest_component_3d(vol_pred)

                # Calculate metrics
                d = dice_coef(vol_gt, vol_pred)
                h = hausdorff_distance_3d(vol_gt, vol_pred)

                dice_scores.append(d)
                hausdorff_scores.append(h)

        mean_dice = np.mean(dice_scores)
        mean_hausdorff = np.mean(hausdorff_scores)

        # Combined Score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
        # We invert Hausdorff because lower is better, but we want to maximize Score.
        # Assuming Hausdorff is normalized 0-1.
        combined_score = (0.4 * mean_dice) + (0.6 * (1.0 - mean_hausdorff))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {avg_train_loss:.6f} | "
            f"Dice: {mean_dice:.6f} | "
            f"HD: {mean_hausdorff:.6f} | "
            f"Score: {combined_score:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save Best Model
        if combined_score > best_score:
            best_score = combined_score
            best_epoch = epoch
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

        scheduler.step()

    print(f"Training complete. Best Score: {best_score:.6f} at Epoch {best_epoch}")
