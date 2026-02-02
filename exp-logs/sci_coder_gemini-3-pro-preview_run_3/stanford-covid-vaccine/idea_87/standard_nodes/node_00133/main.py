import os
import shutil
import warnings
import pandas as pd
import numpy as np
import torch

from library.config import Config
from library.utils import set_seed, metric_mcrmse
from library.data import get_dataloaders
from library.train import run_training
from library.model import RNAModel, generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    set_seed(Config.SEED)

    # Ensure submission directory exists as per requirements
    os.makedirs("./submission", exist_ok=True)

    print("Initializing High-Capacity Stabilized GLU-Decoupled BiGRU Pipeline...")

    # =========================================================================
    # 2. Training
    # =========================================================================
    # Run training for a limited number of epochs for a fast baseline.
    # The dataset is small (1728 samples), so 20 epochs is sufficient and fast.
    print("\n==== Starting Training ====")
    best_mcrmse = run_training(epochs=20, patience=5, debug=False)

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("\n==== Starting Validation & Failure Analysis ====")

    # Load Validation Data
    # We load cached data for speed, but ensure it matches metadata
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    # Note: metric_mcrmse handles slicing to SCORED_COLS and seq_scored internally
    val_metric = metric_mcrmse(all_preds, all_targets)
    print(f"Final Validation Metric: {val_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # 1. Calculate Error Magnitude per Sample
    # Convert to numpy
    preds_np = all_preds.numpy()
    targets_np = all_targets.numpy()

    # Slice to scored length (68) and scored columns
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Slice sequence length
    seq_scored = targets_np.shape[1]
    preds_sliced = preds_np[:, :seq_scored, :]

    # Filter columns
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets_np[:, :, scored_indices]

    # RMSE per sample (averaged over positions and columns)
    # (N, 68, 3) -> (N,)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Load Metadata Features
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Ensure alignment (val_loader is not shuffled)
    if len(val_df) != len(rmse_per_sample):
        print(
            "Warning: Validation dataframe length mismatch. Skipping correlation analysis."
        )
    else:
        val_df["error_magnitude"] = rmse_per_sample

        # Feature Engineering for Analysis
        # GC Content
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s)
        )

        features_to_analyze = ["signal_to_noise", "SN_filter", "gc_content"]

        print("Correlation between Error Magnitude and Input Features:")
        for feat in features_to_analyze:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["error_magnitude"])
                print(f"  {feat}: {corr:.6f}")

    # =========================================================================
    # 4. Submission
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if val_metric < THRESHOLD:
        print(
            f"\nMetric {val_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Generate submission using library function
        generate_submission()

        # Move file to required location
        src_path = Config.SUBMISSION_PATH
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Submission file saved to {dst_path}")
        else:
            print(f"Error: Source submission file not found at {src_path}")
    else:
        print(
            f"\nMetric {val_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
