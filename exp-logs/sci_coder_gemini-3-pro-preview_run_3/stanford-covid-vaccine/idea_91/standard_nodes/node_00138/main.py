import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.train import run_training, generate_submission
from library.model import RNAModel
from library.utils import compute_score, set_seed


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution
    # 20 epochs is sufficient for convergence on this dataset size while fitting in the time limit.
    Config.MAX_EPOCHS = 20

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Running Fast Baseline with Max Epochs: {Config.MAX_EPOCHS}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load cached data if available, otherwise process from metadata
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Training
    # =========================================================================
    # Run the training loop
    run_training(train_loader, val_loader)

    # =========================================================================
    # 4. Validation & Failure Analysis
    # =========================================================================
    print("\nPerforming Validation and Failure Analysis...")
    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Load Best Model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for analysis.")
    else:
        print("Warning: Best model not found. Using current model state.")

    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = model(features, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = compute_score(y_pred, y_true)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n==== Failure Analysis ====")

    # Calculate RMSE per sample on scored columns/positions to match the metric logic
    seq_scored = Config.SEQ_SCORED
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Slice predictions and targets
    y_pred_scored = y_pred[:, :seq_scored, scored_indices]
    y_true_scored = y_true[:, :seq_scored, scored_indices]

    # Mean Squared Error per sample (average over seq_len and channels)
    mse_per_sample = np.mean((y_pred_scored - y_true_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata for Correlation
    val_meta = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = pd.merge(val_meta, error_df, on="id")

    # Feature Engineering for Analysis
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda s: s.count("G") / len(s)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda s: s.count("C") / len(s)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda s: s.count("U") / len(s)
    )

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
    ]

    print("Correlation between Error and Input Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs if any (though data should be clean)
            tmp = analysis_df[[feat, "error"]].dropna()
            if len(tmp) > 1:
                corr, _ = pearsonr(tmp[feat], tmp["error"])
                print(f"  {feat}: {corr:.4f}")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
