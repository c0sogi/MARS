import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided libraries
from library.config import (
    SEED,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    BATCH_SIZE,
    NUM_WORKERS,
    INPUT_DIM,
    PROJECTION_DIM,
    HIDDEN_DIM,
    NUM_LSTM_LAYERS,
    DROPOUT,
    USE_LAYER_NORM,
    SUBMISSION_DIR,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
)
from library.utils import seed_everything, get_device, log_metric, compute_metric
from library.dataset import prepare_data_loaders
from library.model import DPI_BiLSTM
from library.loss_metric import WeightedL1Loss, compute_mae
from library.engine import fit, predict_and_submit


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes the model's errors on the validation set to identify systematic weaknesses.
    Calculates the correlation between absolute error and input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    all_features = []

    # Feature names for correlation mapping
    # Note: The order must match the tensor construction in dataset.py
    # In dataset.py: X is constructed from [CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    feature_names = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)
            if preds.dim() == 3 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            # Calculate absolute error per time step
            # We focus on inspiratory phase (u_out == 0) for the metric,
            # but failure analysis can look at global error dynamics or just inspiratory.
            # Let's look at inspiratory errors specifically as that's the metric.

            abs_error = torch.abs(preds - y)

            # Mask for inspiratory phase
            mask = u_out == 0

            # We need to flatten to correlate
            x_flat = x.cpu().numpy().reshape(-1, len(feature_names))
            err_flat = abs_error.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)

            # Filter by mask
            x_filtered = x_flat[mask_flat]
            err_filtered = err_flat[mask_flat]

            all_errors.append(err_filtered)
            all_features.append(x_filtered)

            # Limit analysis size to avoid OOM on very large val sets if necessary
            # But 20% of data should fit in memory for simple correlation

    # Concatenate
    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(all_features, columns=feature_names)
    df_analysis["abs_error"] = all_errors

    # Calculate correlation
    correlations = (
        df_analysis.corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
    )

    print("Correlation between Input Features and Absolute Error (Inspiratory Phase):")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation
    # We use load_cached_data=True to use preprocessed .npz files if they exist
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        debug=False,  # Ensure we run on full data for valid baseline
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing DPI-BiLSTM model...")
    model = DPI_BiLSTM(
        input_dim=INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        use_layer_norm=USE_LAYER_NORM,
    ).to(device)

    # 4. Training Configuration
    criterion = WeightedL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-6
    )

    # Limit epochs for fast baseline execution as requested
    # Config has 150, but we'll use 25 to ensure < 2 hours runtime
    baseline_epochs = 25

    scheduler = CosineAnnealingLR(optimizer, T_max=baseline_epochs, eta_min=ETA_MIN)

    print(f"Starting training for {baseline_epochs} epochs...")

    # 5. Training Loop
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=baseline_epochs,
        patience=10,  # Early stopping patience
        device=device,
    )

    # 6. Validation Assessment
    print("\nRunning final validation assessment...")
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)
            if preds.dim() == 3 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            batch_mae = compute_mae(preds, y, u_out)
            total_mae += batch_mae
            num_batches += 1

    final_metric = total_mae / num_batches

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    # Threshold from instructions: 0.20567339658737183
    threshold = 0.20567339658737183

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        predict_and_submit(model, test_loader, device, output_path=submission_path)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
