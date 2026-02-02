import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SEED,
    setup_directories,
)
from library.utils import seed_everything, load_model, unscale_target, load_scaler
from library.dataset import VolcanoDataset
from library.model import HybridCRNN
from library.engine import fit, evaluate


def main():
    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    seed_everything(SEED)
    setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fast baseline configuration
    # Limiting epochs to ensure execution finishes quickly (well within 2 hours)
    FAST_EPOCHS = 15

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Initializing Training Dataset...")
    # Initialize train dataset first to compute and save scalers
    train_dataset = VolcanoDataset(
        metadata_path=TRAIN_METADATA_PATH,
        mode="train",
        augment=True,
        load_cached_stats=False,
    )

    print("Initializing Validation Dataset...")
    val_dataset = VolcanoDataset(
        metadata_path=VAL_METADATA_PATH,
        mode="val",
        augment=False,
        load_cached_stats=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    # Determine number of statistical features from the dataset
    # stats_values is a DataFrame in the dataset object
    num_stats_features = train_dataset.stats_values.shape[1]
    print(f"Detected {num_stats_features} statistical features.")

    model = HybridCRNN(num_stats_features=num_stats_features)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Loss function: L1Loss (MAE) on scaled targets
    criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("Starting Training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=FAST_EPOCHS,
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH,
        target_mean=train_dataset.target_mean,
        target_std=train_dataset.target_std,
        scheduler=scheduler,
    )

    # ---------------------------------------------------------
    # 5. Evaluation & Metrics
    # ---------------------------------------------------------
    print("Loading best model for evaluation...")
    model = load_model(model, MODEL_SAVE_PATH, device=device)

    # Compute final metric on validation set
    val_loss, val_mae = evaluate(
        model,
        val_loader,
        criterion,
        device,
        train_dataset.target_mean,
        train_dataset.target_std,
    )

    print(f"Final Validation Metric: {val_mae}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    all_stats = []

    # Collect errors and features from validation set
    with torch.no_grad():
        for spec, stats, target, _ in val_loader:
            spec = spec.to(device)
            stats = stats.to(device)
            target = target.to(device)

            # Predict
            preds = model(spec, stats)

            # Unscale to compute real-world error
            preds_unscaled = unscale_target(
                preds, train_dataset.target_mean, train_dataset.target_std
            )
            target_unscaled = unscale_target(
                target, train_dataset.target_mean, train_dataset.target_std
            )

            # Absolute Error
            error = torch.abs(preds_unscaled - target_unscaled)

            all_errors.append(error.cpu().numpy())
            # Collect input features (normalized stats)
            all_stats.append(stats.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    all_stats = np.concatenate(all_stats, axis=0)  # Shape: (N_samples, N_features)

    # Calculate correlation between Error Magnitude and Feature Values
    feature_names = train_dataset.stats_df.columns
    correlations = []

    for i, feat_name in enumerate(feature_names):
        # Handle potential constant features leading to NaN correlation
        if np.std(all_stats[:, i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(all_stats[:, i], all_errors)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        correlations.append((feat_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Prediction Error:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 2078046.725089643

    if val_mae < THRESHOLD:
        print(
            f"\nValidation metric ({val_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Initialize Test Dataset
        test_dataset = VolcanoDataset(
            metadata_path=TEST_METADATA_PATH,
            mode="test",
            augment=False,
            load_cached_stats=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        segment_ids = []
        predictions = []

        model.eval()
        with torch.no_grad():
            for spec, stats, _, seg_ids in test_loader:
                spec = spec.to(device)
                stats = stats.to(device)

                # Predict
                out = model(spec, stats)

                # Inverse Scale
                out_unscaled = unscale_target(
                    out, train_dataset.target_mean, train_dataset.target_std
                )

                segment_ids.extend(seg_ids.tolist())
                predictions.extend(out_unscaled.cpu().numpy())

        # Create Submission DataFrame
        df_submission = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": predictions}
        )

        # Save
        df_submission.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")

    else:
        print(
            f"\nValidation metric ({val_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
