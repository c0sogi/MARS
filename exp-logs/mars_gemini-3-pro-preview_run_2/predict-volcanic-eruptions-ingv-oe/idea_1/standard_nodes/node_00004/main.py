import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.trainer import run_training, set_seed
from library.data_loader import prepare_data
from library.inference import generate_predictions
from library.feature_extractor import extract_features

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("Initializing pipeline...")
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Computation device: {device}")

    # ---------------------------------------------------------
    # 2. Model Training
    # ---------------------------------------------------------
    # Train the MLP model using the provided trainer module.
    # We use the full dataset (debug_size=None) to ensure the best baseline performance.
    # The trainer handles feature extraction, scaling, and saving the best model.
    print("\n--- Starting Training ---")
    model = run_training(
        debug_size=None,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # ---------------------------------------------------------
    # 3. Validation Assessment
    # ---------------------------------------------------------
    print("\n--- Validation Assessment ---")

    # Reload validation data to perform explicit metric calculation and analysis.
    # prepare_data ensures consistency in scaling.
    _, val_loader, _, _ = prepare_data(
        debug_size=None,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    model.eval()
    val_preds = []
    val_targets = []
    val_inputs = []

    # Efficient inference loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            # Store results
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_inputs.append(inputs.cpu().numpy())

    # Concatenate batches
    y_pred = np.concatenate(val_preds)
    y_true = np.concatenate(val_targets)
    X_val = np.concatenate(val_inputs)

    # Inverse Scale Targets for Metric Calculation
    target_mean_path = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    target_std_path = os.path.join(Config.WORKING_DIR, "target_std.npy")

    if os.path.exists(target_mean_path) and os.path.exists(target_std_path):
        target_mean = np.load(target_mean_path)
        target_std = np.load(target_std_path)
        y_pred = y_pred * target_std + target_mean
        y_true = y_true * target_std + target_mean
        print("Targets inverse scaled for metric calculation.")

    # Compute MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # Print required metric format
    print(f"Final Validation Metric: {mae}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    abs_errors = np.abs(y_pred - y_true)

    # Retrieve feature names to make analysis interpretable
    # We load the dataframe structure from cache/metadata
    df_val_meta = extract_features(
        Config.VAL_METADATA_PATH, Config.VAL_FEATURES_CACHE, load_cached_data=True
    )
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_val_meta.columns if c not in exclude_cols]

    # Create analysis DataFrame
    # X_val is scaled, but correlation is invariant to linear scaling
    analysis_df = pd.DataFrame(X_val, columns=feature_cols)
    analysis_df["abs_error"] = abs_errors

    # Compute correlations between features and error magnitude
    correlations = (
        analysis_df.corr()["abs_error"]
        .drop("abs_error")
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 features correlated with prediction error:")
    print(correlations.head(10))

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    if mae < 5093137.5:
        print("\n--- Generating Submission ---")
        generate_predictions(
            model=model,
            device=device,
            debug_size=None,
            load_cached_data=True,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )
    else:
        print(
            f"\nValidation metric {mae} is not lower than threshold 5093137.5. Skipping submission."
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
