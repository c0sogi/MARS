import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings

# Suppress progress bars from library modules
os.environ["TQDM_DISABLE"] = "1"

# Import provided library modules
from library.config import Config
from library.utils import set_seed, get_device
from library.data import HubmapDataset
from library.model import build_model
from library.train_eval import (
    train_one_epoch,
    validate,
    generate_submission,
    BCEDiceLoss,
    get_anatomical_mask,
)

# --- Configuration Override ---
# Removed Fast Baseline override to allow full training.
# Config.EPOCHS is set in library/config.py


def analyze_failures(model, df_val, device):
    """
    Performs failure analysis on the validation set.
    Calculates Dice per image and correlates with metadata features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    # Setup dataset/loader for analysis (no shuffle, deterministic)
    val_dataset = HubmapDataset(df_val, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Reconstruct predictions for each image
    reconstructed_preds = {}
    for _, row in df_val.iterrows():
        h, w = row["height_pixels"], row["width_pixels"]
        reconstructed_preds[row["id"]] = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            # Standard inference (no TTA) for analysis speed
            preds = torch.sigmoid(model(images)).cpu().numpy()

            for i in range(len(images)):
                img_id = batch["id"][i]
                x = int(batch["x"][i])
                y = int(batch["y"][i])
                pred_tile = preds[i, 0, :, :]

                full_h, full_w = reconstructed_preds[img_id].shape
                h_tile, w_tile = pred_tile.shape

                y_end = min(y + h_tile, full_h)
                x_end = min(x + w_tile, full_w)
                valid_h = y_end - y
                valid_w = x_end - x

                reconstructed_preds[img_id][y:y_end, x:x_end] = pred_tile[
                    :valid_h, :valid_w
                ]

    # Calculate Metrics per Image and Collect Metadata
    results = []
    mask_dir = os.path.join(Config.WORKING_DIR, "masks")

    for _, row in df_val.iterrows():
        img_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]

        # Load Ground Truth
        npy_path = os.path.join(mask_dir, f"{img_id}.npy")
        if os.path.exists(npy_path):
            target_mask = np.load(npy_path)
        else:
            target_mask = np.zeros((h, w), dtype=np.uint8)

        # Apply Anatomical Mask (Cortex)
        cortex_mask = get_anatomical_mask(row["anatomical_json_path"], h, w)
        pred_mask = reconstructed_preds[img_id] * cortex_mask
        pred_binary = (pred_mask > 0.5).astype(np.uint8)

        # Dice Calculation
        intersection = (pred_binary * target_mask).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target_mask.sum() + 1e-7)

        # Collect data
        res = {
            "id": img_id,
            "dice": dice,
            "error": 1.0 - dice,
            "age": row.get("age", np.nan),
            "bmi": row.get("bmi_kg/m^2", np.nan),
            "weight": row.get("weight_kilograms", np.nan),
            "percent_cortex": row.get("percent_cortex", np.nan),
            "percent_medulla": row.get("percent_medulla", np.nan),
        }
        results.append(res)

    results_df = pd.DataFrame(results)

    # Calculate Correlations
    print("Correlation between Error (1-Dice) and Metadata:")
    numeric_cols = ["age", "bmi", "weight", "percent_cortex", "percent_medulla"]
    # Filter columns that exist and have data
    valid_cols = [
        c
        for c in numeric_cols
        if c in results_df.columns and results_df[c].notna().any()
    ]

    if valid_cols:
        corrs = results_df[valid_cols + ["error"]].corr()["error"].drop("error")
        print(corrs)
    else:
        print("Not enough metadata to calculate correlations.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    warnings.filterwarnings("ignore")

    # 2. Data Loading
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Load Datasets
    # We use load_cached_data=True to utilize pre-computed tiles/masks
    train_dataset = HubmapDataset(df_train, mode="train", load_cached_data=True)
    val_dataset = HubmapDataset(df_val, mode="val", load_cached_data=True)

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

    # 3. Model & Optimization
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

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        val_loss, val_dice = validate(model, val_loader, df_val, device, criterion)

        scheduler.step()

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Evaluation & Metric
    # Requirement: Print full precision validation metric
    print(f"Final Validation Metric: {best_dice}")

    # 6. Failure Analysis
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    analyze_failures(model, df_val, device)

    # 7. Conditional Submission
    THRESHOLD = 0.9347
    if best_dice > THRESHOLD:
        generate_submission()
    else:
        # Ensure submission file is not created/overwritten if threshold not met,
        # or handle as per specific logic. Here we just skip calling the generation.
        pass


if __name__ == "__main__":
    main()
