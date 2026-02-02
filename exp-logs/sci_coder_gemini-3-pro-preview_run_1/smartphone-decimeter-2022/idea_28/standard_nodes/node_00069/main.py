import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
import library.config as config
from library.dataset import GnssSequenceDataset, collate_fn
from library.architecture import SEResUNet1D
from library.engine import train_one_epoch, validate
from library.inference import generate_submission
from library.utils import enu_to_geodetic, haversine_distance


def failure_analysis(model, dataloader, device, feature_cols):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()
    all_errors = []
    all_features = []

    # We will sample a subset of points to avoid OOM during correlation calculation if dataset is huge
    # But given the dataset size (~50MB parquet), we can likely process all valid points.

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)  # (B, C, L)
            mask = batch["mask"]  # (B, L)
            targets = batch["targets"]  # (B, 2, L) - ENU targets

            # Forward pass
            outputs = model(features)
            preds_enu = outputs["final"].cpu()  # (B, 2, L)

            # Move features back to CPU for analysis: (B, C, L) -> (B, L, C)
            features_cpu = features.cpu().permute(0, 2, 1)

            # Calculate errors for valid points
            # We approximate error using the ENU targets directly since we want to find
            # correlation with model error magnitude, and ENU distance is the direct training objective.
            # (B, 2, L)
            diff = preds_enu - targets
            # Euclidean distance error in meters
            errors = torch.sqrt(torch.sum(diff**2, dim=1))  # (B, L)

            # Flatten based on mask
            valid_mask = mask.bool()  # (B, L)

            if valid_mask.sum() == 0:
                continue

            batch_errors = errors[valid_mask]  # (N_valid,)
            batch_features = features_cpu[valid_mask]  # (N_valid, C)

            all_errors.append(batch_errors.numpy())
            all_features.append(batch_features.numpy())

    if not all_errors:
        print("No valid data found for failure analysis.")
        return

    # Concatenate
    y_err = np.concatenate(all_errors)
    X_feat = np.concatenate(all_features)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(X_feat, columns=feature_cols)
    df_analysis["Error_Meters"] = y_err

    # Calculate correlation
    correlations = df_analysis.corrwith(df_analysis["Error_Meters"]).drop(
        "Error_Meters"
    )

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print(f"Analyzed {len(y_err)} points.")
    print("Top 10 Features correlated with Error Magnitude:")
    print(top_correlations)
    print("-" * 40)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override config for a fast baseline run
    config.NUM_EPOCHS = 10
    config.BATCH_SIZE = 32
    config.DEBUG = False  # Use full data provided in working directory

    device = config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing Datasets...")
    try:
        # Load cached data if available to save time
        train_dataset = GnssSequenceDataset(split="train", load_cached_data=True)
        val_dataset = GnssSequenceDataset(split="val", load_cached_data=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Ensure metadata and processed parquet files exist.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = SEResUNet1D().to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training...")
    best_metric = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        # Note: validate function prints metric internally
        val_metric = validate(model, val_loader, device)

        scheduler.step()

        # Save Best Model
        if val_metric < best_metric:
            best_metric = val_metric
            print(f"New best metric: {best_metric:.4f}. Saving model...")
            torch.save(model.state_dict(), config.MODEL_PATH)

    print(f"\nTraining Complete. Best Validation Metric: {best_metric:.9f}")

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("\nRunning Final Validation Assessment...")
    # Load best model weights
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    failure_analysis(model, val_loader, device, train_dataset.feature_cols)

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 3.7864967500302016

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        try:
            generate_submission(load_cached_data=True)
            print(f"Submission saved to {config.SUBMISSION_PATH}")
        except Exception as e:
            print(f"Error generating submission: {e}")
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
