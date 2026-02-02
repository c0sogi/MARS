import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.train import run_training, generate_submission
from library.data import get_loaders
from library.model import CEAMSDS
from library.utils import expm1_transform, rmsle, set_seed


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("============================================================")
    print(" CEA-MS-DS Pipeline Execution")
    print("============================================================")

    # 1. Train the model
    # We use the full dataset (debug_size=None) as it is small (1728 samples).
    # We reduce epochs to 100 to ensure quick baseline execution while allowing convergence.
    print("\n[Step 1] Starting Model Training...")
    run_training(debug_size=None, num_epochs=100)

    # 2. Load Best Model and Validation Data
    print("\n[Step 2] Loading Best Model for Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    model = CEAMSDS().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Get loaders (we only need val_loader here, but get_loaders returns all)
    # load_cached_scalers=True ensures we use the scalers fitted during training
    _, val_loader, _ = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_scalers=True
    )

    # 3. Validation Inference
    print("\n[Step 3] Running Inference on Validation Set...")
    val_preds_log = []
    val_targets_log = []

    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for batch in val_loader:
            x_atomic, x_global, y, batch_indices = batch

            # Move to device
            x_atomic = x_atomic.to(device)
            x_global = x_global.to(device)
            y = y.to(device)
            batch_indices = batch_indices.to(device)

            # Forward pass
            outputs = model(x_atomic, x_global, batch_indices)

            # Collect results (move to CPU)
            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(y.cpu().numpy())

    # Concatenate all batches
    val_preds_log = np.concatenate(val_preds_log, axis=0)
    val_targets_log = np.concatenate(val_targets_log, axis=0)

    # Inverse transform targets and predictions (log1p -> expm1) to get original scale
    val_preds = expm1_transform(val_preds_log)
    val_targets = expm1_transform(val_targets_log)

    # Clip negative predictions to 0 (physical constraint)
    val_preds = np.maximum(val_preds, 0)
    val_targets = np.maximum(val_targets, 0)

    # 4. Compute Metric
    # The rmsle function applies log1p internally, so we pass original scale values
    score = rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    print("\n[Step 4] Performing Failure Analysis...")

    # Calculate error magnitude per sample
    # We use Mean Squared Error in log space as a proxy for "badness" of prediction relative to the metric
    # Error = mean((log_pred - log_true)^2) across the two targets
    sample_errors = np.mean((val_preds_log - val_targets_log) ** 2, axis=1)

    # Load original validation metadata to correlate errors with input features
    if os.path.exists(Config.VAL_CSV):
        val_df = pd.read_csv(Config.VAL_CSV)

        # Ensure lengths match (DataLoader preserves order if shuffle=False)
        if len(val_df) == len(sample_errors):
            val_df["model_error"] = sample_errors

            # Select numeric columns for correlation analysis
            numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude ID, targets, and the error column itself from features list
            cols_to_exclude = ["id", "model_error"] + Config.TARGET_COLS
            feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]

            # Compute correlations
            correlations = (
                val_df[feature_cols]
                .corrwith(val_df["model_error"])
                .abs()
                .sort_values(ascending=False)
            )

            print("Top 10 features correlated with model error (absolute correlation):")
            print(correlations.head(10))
        else:
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) matches prediction length ({len(sample_errors)}) mismatch. Skipping correlation analysis."
            )
    else:
        print("Validation metadata file not found. Skipping failure analysis.")

    # 6. Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 0.04819517582654953

    print("\n[Step 5] Submission Check...")
    if score < THRESHOLD:
        print(f"Validation score ({score:.6f}) meets threshold ({THRESHOLD:.6f}).")
        print("Generating submission file...")
        generate_submission(model=model, device=device)
    else:
        print(
            f"Validation score ({score:.6f}) does NOT meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
