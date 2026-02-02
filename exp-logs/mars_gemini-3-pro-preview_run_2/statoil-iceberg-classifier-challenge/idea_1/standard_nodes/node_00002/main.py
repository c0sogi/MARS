import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

# Import from the provided library files
from library.config import Config
from library.model import D2N, train_model, generate_submission
from library.data_loader import get_data_loaders


def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run():
    # 1. Setup
    set_seeds(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Using cached data if available for speed
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = D2N(
        input_dim=Config.INPUT_DIM,
        hidden_units=Config.HIDDEN_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 4. Training
    print("Starting training...")
    # train_model handles the loop, early stopping, and returns the best model state
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        lr=Config.LEARNING_RATE,
    )

    # 5. Validation Assessment & Failure Analysis
    print("Performing validation assessment...")
    model.eval()

    val_preds = []
    val_targets = []
    val_inputs = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
            val_inputs.append(X_batch.cpu().numpy())

    # Concatenate batches
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_inputs = np.concatenate(val_inputs, axis=0)

    # Calculate Metric (Log Loss)
    # Clip predictions to avoid log(0) errors, though Sigmoid usually handles this
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    metric = log_loss(val_targets, val_preds_clipped)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Extract features for correlation analysis
    # Structure: [pixel_0, ..., pixel_N, inc_angle]
    # Note: Inputs are scaled (StandardScaler), but correlations are scale-invariant

    # Last column is incidence angle
    inc_angles = val_inputs[:, -1]

    # All other columns are image pixels
    image_pixels = val_inputs[:, :-1]
    img_means = np.mean(image_pixels, axis=1)
    img_stds = np.std(image_pixels, axis=1)

    # Create a DataFrame for correlation calculation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_angles,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Generate Submission
    baseline_metric = 0.6780661922590535
    if metric < baseline_metric:
        print("\nGenerating submission...")
        generate_submission(model, test_loader, test_ids)
    else:
        print(
            f"\nValidation metric {metric} did not improve upon baseline {baseline_metric}. Skipping submission."
        )

    print("Pipeline complete.")


if __name__ == "__main__":
    run()
