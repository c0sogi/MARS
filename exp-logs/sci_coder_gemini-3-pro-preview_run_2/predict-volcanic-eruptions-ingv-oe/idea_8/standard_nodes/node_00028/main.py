import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, TargetScaler, load_checkpoint
from library.data_loader import get_dataloaders
from library.model import SeismicHybridModel
from library.engine import run_training, predict_fn


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    # Reducing epochs to ensure completion within time limits
    Config.EPOCHS = 15
    print(f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}")

    # 2. Data Loading
    # load_cached_data=True will use the parquet files generated in ./working
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=True
    )

    # Determine input dimension for the MLP branch
    # Get one batch to inspect shape
    sample_batch = next(iter(train_loader))
    num_tabular_features = sample_batch["tabular"].shape[1]
    print(f"Detected {num_tabular_features} tabular features.")

    # 3. Model Initialization
    model = SeismicHybridModel(num_tabular_features=num_tabular_features)
    model.to(Config.DEVICE)

    # 4. Training
    # run_training handles the loop, validation, and saving the best checkpoint
    run_training(model, train_loader, val_loader, Config.DEVICE, target_scaler)

    # 5. Validation Assessment & Failure Analysis
    print("\nRunning Validation Assessment...")

    # Load best model
    model = load_checkpoint(model, Config.MODEL_PATH, Config.DEVICE)
    model.eval()

    val_preds = []
    val_targets = []
    val_segment_ids = []
    val_features_list = []

    # Manual inference loop on validation set to gather all data for analysis
    with torch.no_grad():
        for data in val_loader:
            spectrogram = data["spectrogram"].to(Config.DEVICE)
            tabular = data["tabular"].to(Config.DEVICE)
            targets = data["target"].to(Config.DEVICE)
            ids = data["segment_id"]

            outputs = model(spectrogram, tabular)
            outputs = outputs.squeeze(1)

            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
            val_segment_ids.extend(ids.numpy())
            val_features_list.extend(tabular.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_features = np.array(val_features_list)

    # Inverse transform to get original scale
    val_preds_unscaled = target_scaler.inverse_transform(val_preds)
    val_targets_unscaled = target_scaler.inverse_transform(val_targets)

    # Compute Metric
    mae = np.mean(np.abs(val_preds_unscaled - val_targets_unscaled))
    print(f"Final Validation Metric: {mae}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    absolute_errors = np.abs(val_preds_unscaled - val_targets_unscaled)

    # We need feature names to make sense of correlations.
    # We can retrieve them from the cached parquet file or reconstruct generic names.
    # Since we don't have the column names handy in the loader, we'll load the parquet columns briefly.
    try:
        val_df = pd.read_parquet(Config.VAL_FEATURES_PATH)
        # Exclude segment_id
        feature_names = [c for c in val_df.columns if c != "segment_id"]
    except:
        feature_names = [f"feat_{i}" for i in range(num_tabular_features)]

    # Calculate correlation between Error and Features
    # Create a DataFrame for easy correlation computation
    analysis_df = pd.DataFrame(val_features, columns=feature_names)
    analysis_df["abs_error"] = absolute_errors

    correlations = (
        analysis_df.corr()["abs_error"]
        .drop("abs_error")
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Absolute Error:")
    print(correlations.head(5))

    # 6. Submission
    THRESHOLD = 1492505.6322055138

    if mae < THRESHOLD:
        print(
            f"\nValidation MAE ({mae:.4f}) is better than threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        # Inference on Test Set
        segment_ids, test_preds = predict_fn(test_loader, model, Config.DEVICE)

        # Inverse transform
        test_preds_unscaled = target_scaler.inverse_transform(test_preds)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "segment_id": segment_ids.astype(int),
                "time_to_eruption": test_preds_unscaled,
            }
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())

    else:
        print(
            f"\nValidation MAE ({mae:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
