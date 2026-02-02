import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import rasterio.features
from tqdm.auto import tqdm

from library.config import Config
from library.utils import set_seed, get_device, rle_encode, rle_decode
from library.data import HubmapDataset, get_cortex_polygons
from library.model import build_model

# --- Loss Function ---


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Dice Loss
        preds_sigmoid = torch.sigmoid(preds)
        smooth = 1.0

        # Flatten
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (
            preds_flat.sum() + targets_flat.sum() + smooth
        )
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# --- Helper Functions ---


def get_anatomical_mask(anatomical_json_path, height, width):
    """
    Generates a binary mask for the Cortex region from the anatomical JSON.
    Returns a mask of ones if no cortex polygon is found (fallback).
    """
    poly = get_cortex_polygons(anatomical_json_path)
    if poly is None:
        return np.ones((height, width), dtype=np.uint8)

    # Rasterize the polygon
    mask = rasterio.features.rasterize(
        [poly], out_shape=(height, width), fill=0, default_value=1, dtype=np.uint8
    )
    return mask


# --- Training and Validation Functions ---


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per iteration if it's not ReduceLROnPlateau
        # Config uses CosineAnnealingLR which is typically stepped per epoch,
        # but can be per step. Here we assume per epoch logic in the main loop,
        # but if T_MAX is based on epochs, we step outside.
        # However, standard practice for Cosine is often per epoch.
        # We will step scheduler in the main loop.

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def validate(model, loader, df_val, device, criterion):
    model.eval()
    val_loss = 0.0

    # Dictionary to reconstruct images: {id: np.array}
    reconstructed_preds = {}
    reconstructed_targets = {}

    # Initialize buffers
    for _, row in df_val.iterrows():
        h, w = row["height_pixels"], row["width_pixels"]
        reconstructed_preds[row["id"]] = np.zeros((h, w), dtype=np.float32)
        # We will load targets from .npy files directly for efficiency later

    # Inference on tiles
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

            preds_prob = torch.sigmoid(outputs).cpu().numpy()

            # Place tiles back into full image
            # Batch size might be > 1
            for i in range(len(images)):
                img_id = batch["id"][i]
                x = int(batch["x"][i])
                y = int(batch["y"][i])
                pred_tile = preds_prob[i, 0, :, :]  # (H, W)

                # Handle boundaries
                full_h, full_w = reconstructed_preds[img_id].shape
                h_tile, w_tile = pred_tile.shape

                y_end = min(y + h_tile, full_h)
                x_end = min(x + w_tile, full_w)

                # Crop tile if it extends beyond image (due to padding in dataset)
                valid_h = y_end - y
                valid_w = x_end - x

                reconstructed_preds[img_id][y:y_end, x:x_end] = pred_tile[
                    :valid_h, :valid_w
                ]

    # Calculate Global Dice with Anatomical Filtering
    dice_scores = []
    mask_dir = os.path.join(Config.WORKING_DIR, "masks")

    for _, row in df_val.iterrows():
        img_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]

        # Load Ground Truth
        npy_path = os.path.join(mask_dir, f"{img_id}.npy")
        if os.path.exists(npy_path):
            target_mask = np.load(npy_path)
        else:
            # Fallback if not cached (should not happen with proper pipeline)
            target_mask = np.zeros((h, w), dtype=np.uint8)

        # Get Anatomical Mask (Cortex)
        cortex_mask = get_anatomical_mask(row["anatomical_json_path"], h, w)

        # Apply filtering
        pred_mask = reconstructed_preds[img_id] * cortex_mask

        # Threshold
        pred_binary = (pred_mask > 0.5).astype(np.uint8)

        # Dice Calculation
        intersection = (pred_binary * target_mask).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target_mask.sum() + 1e-7)
        dice_scores.append(dice)

    avg_dice = np.mean(dice_scores)
    avg_loss = val_loss / len(loader)

    return avg_loss, avg_dice


def run_training():
    set_seed(Config.SEED)
    device = get_device()

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Datasets & Loaders
    train_dataset = HubmapDataset(df_train, mode="train")
    val_dataset = HubmapDataset(df_val, mode="val")

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
    )

    # Model, Optimizer, Scheduler, Loss
    model = build_model().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )
    criterion = BCEDiceLoss()
    scaler = GradScaler()

    best_dice = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        val_loss, val_dice = validate(model, val_loader, df_val, device, criterion)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice: {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Dice! Model saved to {Config.BEST_MODEL_PATH}")

    print(f"Training Complete. Best Validation Dice: {best_dice:.6f}")


# --- Inference Function ---


def generate_submission():
    set_seed(Config.SEED)
    device = get_device()

    # Load Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load Model
    model = build_model()
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Best model not found. Using random weights (for debugging/testing flow)."
        )

    model.to(device)
    model.eval()

    # Dataset & Loader
    test_dataset = HubmapDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Reconstruction buffer
    reconstructed_preds = {}
    for _, row in df_test.iterrows():
        h, w = row["height_pixels"], row["width_pixels"]
        reconstructed_preds[row["id"]] = np.zeros((h, w), dtype=np.float32)

    print("Running Inference with TTA...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)

            # --- 4-Way TTA ---
            # 1. Original
            pred_1 = torch.sigmoid(model(images))

            # 2. Horizontal Flip
            pred_2 = torch.sigmoid(model(torch.flip(images, dims=[3])))
            pred_2 = torch.flip(pred_2, dims=[3])

            # 3. Vertical Flip
            pred_3 = torch.sigmoid(model(torch.flip(images, dims=[2])))
            pred_3 = torch.flip(pred_3, dims=[2])

            # 4. Rotate 90
            pred_4 = torch.sigmoid(model(torch.rot90(images, k=1, dims=[2, 3])))
            pred_4 = torch.rot90(pred_4, k=-1, dims=[2, 3])

            # Average
            avg_pred = (pred_1 + pred_2 + pred_3 + pred_4) / 4.0
            preds_prob = avg_pred.cpu().numpy()

            # Reconstruct
            for i in range(len(images)):
                img_id = batch["id"][i]
                x = int(batch["x"][i])
                y = int(batch["y"][i])
                pred_tile = preds_prob[i, 0, :, :]

                full_h, full_w = reconstructed_preds[img_id].shape
                h_tile, w_tile = pred_tile.shape

                y_end = min(y + h_tile, full_h)
                x_end = min(x + w_tile, full_w)

                valid_h = y_end - y
                valid_w = x_end - x

                reconstructed_preds[img_id][y:y_end, x:x_end] = pred_tile[
                    :valid_h, :valid_w
                ]

    # Post-processing and Encoding
    submission_rows = []
    print("Encoding predictions...")

    for _, row in df_test.iterrows():
        img_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]

        # Get Anatomical Mask
        cortex_mask = get_anatomical_mask(row["anatomical_json_path"], h, w)

        # Filter and Threshold
        final_mask = (reconstructed_preds[img_id] * cortex_mask > 0.5).astype(np.uint8)

        # RLE Encode
        rle = rle_encode(final_mask)
        submission_rows.append({"id": img_id, "predicted": rle})

        # Clean up memory
        del reconstructed_preds[img_id]
        gc.collect()

    # Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
