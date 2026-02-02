import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.train import train_loop
from library.inference import make_predictions
from library.dataset import get_train_val_loaders
from library.model import ConvNeXtUNet


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    print("Configuring for fast baseline run...")
    # Limit training data and epochs to ensure completion within 2 hours
    Config.MAX_TRAIN_SAMPLES = 3000
    Config.EPOCHS = 5
    # Limit validation during training loop for speed
    Config.MAX_VAL_SAMPLES = 500

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Model Training
    # ==========================================
    print(
        f"Starting training with {Config.MAX_TRAIN_SAMPLES} samples for {Config.EPOCHS} epochs..."
    )
    # train_loop handles data loading, model init, training, and saving best_model.pth
    train_loop(epochs=Config.EPOCHS, patience=3)

    # ==========================================
    # 3. Full Validation Assessment
    # ==========================================
    print("Performing full validation assessment...")
    device = Config.DEVICE

    # Reset to None to load the FULL validation set for final scoring
    Config.MAX_VAL_SAMPLES = None

    # We only need the validation loader here
    _, val_loader = get_train_val_loaders()

    # Load the best model weights
    model = ConvNeXtUNet()
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model weights not found.")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Metrics accumulators
    total_intersection = 0.0
    total_union = 0.0

    # Failure Analysis accumulators
    analysis_records = []
    val_df = val_loader.dataset.df  # Access underlying dataframe for metadata

    print("Running inference on full validation set...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Inference with mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Binarize
            preds = (probs > Config.THRESHOLD).float()

            # --- Global Dice Calculation ---
            # Flatten batch for global intersection/union
            preds_flat = preds.view(preds.size(0), -1)
            targets_flat = masks.view(masks.size(0), -1)

            intersection = (preds_flat * targets_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

            total_intersection += intersection.sum().item()
            total_union += union.sum().item()

            # --- Failure Analysis Data Collection ---
            # Compute per-sample Dice
            smooth = 1e-6
            sample_dices = (2.0 * intersection + smooth) / (union + smooth)
            sample_dices = sample_dices.cpu().numpy()

            # Compute simple image statistic: Mean of Channel 2 (Temperature Proxy)
            # Input is (B, 6, H, W). Channel 2 is Band 14 (Temp).
            img_means = images[:, 2, :, :].mean(dim=(1, 2)).cpu().numpy()

            # Map back to metadata
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_df = val_df.iloc[start_idx:end_idx]

            for j, (_, row) in enumerate(batch_df.iterrows()):
                analysis_records.append(
                    {
                        "dice": sample_dices[j],
                        "img_mean_temp": img_means[j],
                        "timestamp": row.get("timestamp", np.nan),
                        "row_min": row.get("row_min", np.nan),
                        "col_min": row.get("col_min", np.nan),
                    }
                )

    # Compute Final Global Dice
    smooth = 1e-6
    global_dice = (2.0 * total_intersection + smooth) / (total_union + smooth)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {global_dice}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming failure analysis...")
    analysis_df = pd.DataFrame(analysis_records)
    analysis_df["error"] = 1.0 - analysis_df["dice"]

    features_to_check = ["img_mean_temp", "timestamp", "row_min", "col_min"]
    print("Correlation between Error (1-Dice) and features:")

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs for correlation calculation
            tmp_df = analysis_df.dropna(subset=[feat, "error"])
            if len(tmp_df) > 1:
                # Calculate Pearson correlation
                corr, _ = pearsonr(tmp_df[feat], tmp_df["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Insufficient data")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD_SCORE = 0.5910660985501295

    if global_dice > THRESHOLD_SCORE:
        print(
            f"\nMetric ({global_dice:.6f}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        make_predictions()
    else:
        print(
            f"\nMetric ({global_dice:.6f}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
