import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader

from library.config import Config
from library.train import run_training, run_inference
from library.data import get_dataset, RNADataset
from library.modules import AHCHDN
from library.utils import seed_everything, calculate_global_rmse


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # We use the full dataset (debug=False) because the dataset size (approx 2k samples)
    # is small enough to train 15 epochs quickly on an A100, ensuring we meet the
    # performance threshold.
    print("Starting training pipeline...")
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Validation & Metric Calculation
    print("\nStarting validation analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best model
    model = AHCHDN().to(device)
    model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Error: Best model not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load validation data
    val_data = get_dataset("val", load_cached_data=True)
    val_ds = RNADataset(val_data, "val")
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Inference on validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_map, targets in val_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)

            # Two-pass inference strategy
            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    metric = calculate_global_rmse(
        all_preds,
        all_targets,
        scored_length=Config.SCORED_LENGTH,
        scored_cols_indices=Config.SCORED_COLS_INDICES,
    )
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming failure analysis...")

    # Calculate RMSE per sample (scalar value for correlation)
    # Slice to scored region and columns
    preds_scored = all_preds[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]
    targs_scored = all_targets[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]

    # MSE per sample: average over sequence length (axis 1) and columns (axis 2)
    mse_per_sample = np.mean((preds_scored - targs_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to correlate
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Ensure alignment (dataset loader preserves order of csv)
    if len(val_df) != len(rmse_per_sample):
        print("Warning: Mismatch in validation set size for analysis.")
    else:
        # Correlation with Signal to Noise
        if "signal_to_noise" in val_df.columns:
            sn = val_df["signal_to_noise"].values
            corr_sn, _ = stats.pearsonr(rmse_per_sample, sn)
            print(f"Correlation (Error vs Signal_to_Noise): {corr_sn}")

        # Correlation with Mean Reactivity
        if "mean_reactivity" in val_df.columns:
            mr = val_df["mean_reactivity"].values
            corr_mr, _ = stats.pearsonr(rmse_per_sample, mr)
            print(f"Correlation (Error vs Mean_Reactivity): {corr_mr}")

        # Correlation with Sequence Length (though constant 107, good check)
        if "seq_length" in val_df.columns:
            sl = val_df["seq_length"].values
            if np.std(sl) > 0:
                corr_sl, _ = stats.pearsonr(rmse_per_sample, sl)
                print(f"Correlation (Error vs Seq_Length): {corr_sl}")

    # 5. Submission
    threshold = 0.47142532743789534
    if metric < threshold:
        print(f"\nMetric {metric} < threshold {threshold}. Generating submission...")
        run_inference(batch_size=Config.BATCH_SIZE)
    else:
        print(f"\nMetric {metric} >= threshold {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
