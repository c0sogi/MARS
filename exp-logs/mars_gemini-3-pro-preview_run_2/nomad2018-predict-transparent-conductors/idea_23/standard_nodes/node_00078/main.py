import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.data import get_dataloaders
from library.model import RA_CGN, train_one_epoch, predict
from library.train import evaluate, generate_submission
from library.utils import set_seed


def perform_failure_analysis(model, val_loader, device, scaler, metadata_path):
    """
    Analyzes the correlation between prediction errors and input features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Get predictions on validation set
    val_preds, val_ids = predict(model, val_loader, device, scaler)

    # Load metadata
    val_meta = pd.read_csv(metadata_path)

    # Ensure alignment
    val_meta = val_meta.set_index("id").loc[val_ids].reset_index()

    # Get targets
    targets = val_meta[Config.TARGET_COLS].values

    # Compute error magnitude (Mean Absolute Error per sample across targets)
    # We use MAE as a proxy for "error magnitude"
    errors = np.mean(np.abs(val_preds - targets), axis=1)

    # Select numerical features for correlation
    feature_cols = [
        col
        for col in val_meta.columns
        if col not in ["id", "file_path"] + Config.TARGET_COLS
        and pd.api.types.is_numeric_dtype(val_meta[col])
    ]

    print(f"Correlating Mean Absolute Error with {len(feature_cols)} features...")

    correlations = {}
    for col in feature_cols:
        if val_meta[col].nunique() > 1:
            corr, _ = pearsonr(val_meta[col], errors)
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features associated with Error:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat:<30}: {corr:+.4f}")
    print("=" * 40 + "\n")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Override Config for fast baseline execution while maintaining performance
    # We use the full dataset but a reasonable number of epochs for convergence
    Config.EPOCHS = 150
    Config.DEBUG_SAMPLE_SIZE = None  # Use full data

    # 2. Data Loading
    # load_cached_data=True utilizes the preprocessed .npz files in ./working/idea_23/cache/
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = RA_CGN(Config).to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_val_rmsle = float("inf")
    patience_counter = 0

    print(f"\nStarting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, device, scaler)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSLE: {val_rmsle:.6f}"
        )

        # Checkpointing
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute final metric on validation set
    _, final_metric = evaluate(model, val_loader, criterion, device, scaler)
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    perform_failure_analysis(
        model, val_loader, device, scaler, Config.VAL_METADATA_PATH
    )

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.049412816762924194

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_metric} does NOT meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
