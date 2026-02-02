import pandas as pd
import numpy as np
import torch
import sys
import os

# Import from provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import get_dataloaders
from library.model import DGC_BiLSTM
from library.train import run_training

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure completion within ~2 hours while maintaining convergence quality
Config.EPOCHS = 30
# Increase batch size to utilize A100 GPU memory and speed up training
Config.BATCH_SIZE = 512


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("==========================================")
    print(" Starting Fast Baseline Execution ")
    print(f" Epochs: {Config.EPOCHS}")
    print(f" Batch Size: {Config.BATCH_SIZE}")
    print("==========================================\n")

    # 1. Run Training
    # This handles data loading (with caching), model init, training loop, and saving best model.
    run_training(debug=False)

    print("\n==========================================")
    print(" Training Complete. Starting Evaluation. ")
    print("==========================================\n")

    # 2. Load Best Model
    device = torch.device(Config.DEVICE)
    model = DGC_BiLSTM().to(device)

    checkpoint = load_checkpoint(model, path=Config.BEST_MODEL_PATH)
    if checkpoint is None:
        print("Error: No checkpoint found at", Config.BEST_MODEL_PATH)
        return

    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with Val MAE: {checkpoint['best_mae']:.6f}"
    )

    # 3. Validation Inference & Metric Calculation
    # We reload loaders. load_cached_data=True ensures we use the exact same preprocessed data.
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_features = []

    # Feature names corresponding to the 12 input features in data_loader.py
    feature_names = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
    ]

    print("Running validation inference...")
    with torch.no_grad():
        for features, targets, u_out in val_loader:
            features = features.to(device)

            # Forward pass
            preds = model(features)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.numpy())
            val_u_out.append(u_out.numpy())
            val_features.append(features.cpu().numpy())

    # Flatten sequences for analysis
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_u_out = np.concatenate(val_u_out).flatten()
    val_features = np.concatenate(val_features).reshape(-1, len(feature_names))

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    insp_mask = val_u_out == 0
    abs_errors = np.abs(val_preds - val_targets)
    insp_errors = abs_errors[insp_mask]

    final_metric = np.mean(insp_errors)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis (Inspiratory Phase) ---")

    # Filter features to inspiratory phase
    insp_features = val_features[insp_mask]

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(insp_features, columns=feature_names)
    df_analysis["error_magnitude"] = insp_errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    THRESHOLD = 0.1619843989610672

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} is below threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []

        print("Running test inference...")
        with torch.no_grad():
            for features, _, _ in test_loader:
                features = features.to(device)
                preds = model(features)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds).flatten()

        # Load the processed test dataframe to ensure ID alignment
        # The test_loader iterates over Config.TEST_CACHE (parquet), so we load that to get IDs.
        if os.path.exists(Config.TEST_CACHE):
            df_test_processed = pd.read_parquet(Config.TEST_CACHE)
            # Ensure sorted exactly as the loader processed it (though parquet read usually preserves order)
            df_test_processed = df_test_processed.sort_values(
                ["breath_id", "time_step"]
            )
            submission_ids = df_test_processed["id"].values
        else:
            # Fallback (should not happen if pipeline ran)
            print(
                "Warning: Test cache not found. Using metadata (risk of misalignment if not sorted)."
            )
            test_meta = pd.read_csv(Config.TEST_METADATA)
            submission_ids = test_meta["id"].values

        if len(test_preds) != len(submission_ids):
            raise ValueError(
                f"Shape mismatch: Preds {len(test_preds)} vs IDs {len(submission_ids)}"
            )

        submission = pd.DataFrame({"id": submission_ids, "pressure": test_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric:.6f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
