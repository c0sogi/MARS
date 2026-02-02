import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided libraries
import library.config as C
import library.utils as U
from library.dataset import get_datasets, gnss_collate_fn
from library.model import TransResUNet
from library.engine import Trainer, generate_submission


def calculate_competition_metric(errors):
    """
    Computes the mean of the 50th and 95th percentile errors.
    """
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)
    return (p50 + p95) / 2


def run_validation_and_analysis(model, val_loader, device, feature_cols):
    """
    Runs inference on validation set, computes competition metric,
    and performs failure analysis (correlation).
    """
    model.eval()

    trip_scores = []
    all_errors = []
    all_features = []

    print("\nRunning Validation and Failure Analysis...")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # (B, L, C)
            targets = batch["targets"].to(device)  # (B, L, 2)
            phone_idx = batch["phone_idx"].to(device)
            lengths = batch["lengths"]

            # Forward pass
            # Model expects (B, C, L)
            features_in = features.permute(0, 2, 1)
            outputs = model(features_in, phone_idx)  # (B, 2, L)
            outputs = outputs.permute(0, 2, 1)  # (B, L, 2)

            # Process each sequence in the batch
            batch_size = features.size(0)
            for i in range(batch_size):
                length = lengths[i]

                # Extract valid sequence data
                pred_seq = outputs[i, :length, :].cpu().numpy()
                target_seq = targets[i, :length, :].cpu().numpy()
                feat_seq = features[i, :length, :].cpu().numpy()

                # Calculate Euclidean distance in ENU space (Meters)
                # target is dLat_meters (North), dLon_meters (East)
                # pred is same
                diff = pred_seq - target_seq
                # dist = sqrt(dNorth^2 + dEast^2)
                errors = np.sqrt(np.sum(diff**2, axis=1))

                # Metric for this trip
                score = calculate_competition_metric(errors)
                trip_scores.append(score)

                # Collect for failure analysis
                all_errors.append(errors)
                all_features.append(feat_seq)

    # 1. Compute Final Metric
    final_metric = np.mean(trip_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("\n--- Failure Analysis ---")
    flat_errors = np.concatenate(all_errors)
    flat_features = np.concatenate(all_features)

    # Compute correlation between error magnitude and each feature
    correlations = {}
    for idx, col_name in enumerate(feature_cols):
        # Handle constant columns to avoid NaN correlation
        feat_vals = flat_features[:, idx]
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, flat_errors)[0, 1]
        correlations[col_name] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features:")
    for name, corr in sorted_corr:
        print(f"  {name}: {corr:.4f}")

    return final_metric


def main():
    # 1. Setup
    C.set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Limit training drives for fast baseline, but use full validation/test
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True, max_drives=10)

    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences:   {len(val_ds)}")
    print(f"Test sequences:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=C.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = TransResUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY
    )

    # 4. Training
    # Using reduced epochs for fast baseline verification as per instructions
    trainer = Trainer(model, train_loader, val_loader, optimizer, device)
    trainer.fit(epochs=5, save_path=os.path.join(C.WORKING_DIR, "model_weights.pth"))

    # 5. Validation & Analysis
    # Load best model
    model.load_state_dict(torch.load(os.path.join(C.WORKING_DIR, "model_weights.pth")))

    feature_cols = train_ds.feature_columns
    val_metric = run_validation_and_analysis(model, val_loader, device, feature_cols)

    # 6. Submission
    threshold = 3.802240262877392
    if val_metric < threshold:
        print(
            f"\nValidation metric {val_metric} < {threshold}. Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(f"\nValidation metric {val_metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
