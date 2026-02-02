import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.data import load_data, collate_fn
from library.model import DeepDecoupledBiGRU
from library.train import run_training, generate_submission
from library.utils import seed_everything, mcrmse_loss


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for specific task requirements
    # Ensure submission directory exists
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.MAX_EPOCHS}")
    print(f"  Submission File: {Config.SUBMISSION_FILE}")

    # ==========================================
    # 2. Training
    # ==========================================
    # Run the training pipeline provided in library.train
    # This handles data loading, model training, and saving the best checkpoint.
    print("\n=== Starting Training Pipeline ===")
    run_training()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load Validation Data
    val_dataset = load_data("val", load_cached_data=True, debug=Config.DEBUG)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Load Best Model
    model = DeepDecoupledBiGRU().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(Config.DEVICE)
            pair_indices = batch["pair_indices"].to(Config.DEVICE)
            targets = batch["targets"].to(Config.DEVICE)
            ids = batch["id"]

            preds = model(features, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Official Metric
    final_metric = mcrmse_loss(all_targets, all_preds).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # 1. Calculate Error per Sample
    # We follow the metric logic: Slice to 68, Filter columns [0, 1, 3]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    y_true_sliced = all_targets[:, : Config.PRED_LEN, scored_indices]
    y_pred_sliced = all_preds[:, : Config.PRED_LEN, scored_indices]

    # MSE per sample (average over sequence and channels)
    # Shape: (N, 68, 3) -> (N,)
    mse_per_sample = torch.mean((y_true_sliced - y_pred_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 2. Load Metadata to correlate
    val_df = pd.read_parquet(Config.VAL_PARQUET)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Calculate Correlations
    correlations = {}
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]

    # Add nucleotide content features for analysis
    if "sequence" in analysis_df.columns:
        analysis_df["pct_A"] = analysis_df["sequence"].apply(
            lambda s: s.count("A") / len(s)
        )
        analysis_df["pct_G"] = analysis_df["sequence"].apply(
            lambda s: s.count("G") / len(s)
        )
        analysis_df["pct_U"] = analysis_df["sequence"].apply(
            lambda s: s.count("U") / len(s)
        )
        features_to_check.extend(["pct_A", "pct_G", "pct_U"])

    print("Correlation between Model Error (RMSE) and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs just in case
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feat].corr(valid_data["error"])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()

        # Verify file creation
        if os.path.exists(Config.SUBMISSION_FILE):
            print(f"Submission successfully created at: {Config.SUBMISSION_FILE}")
        else:
            print("Error: Submission file was not found after generation.")
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
