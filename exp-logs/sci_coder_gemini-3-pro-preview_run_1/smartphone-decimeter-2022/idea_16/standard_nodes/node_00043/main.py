import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_dataset, GNSSSequenceDataset
from library.model import ResUNet1D
from library.train import train_model
from library.inference import generate_predictions


def run_validation_and_analysis():
    """
    Runs inference on the validation set, computes the official competition metric,
    and performs failure analysis by correlating errors with input features.
    """
    print("\n" + "=" * 40)
    print("STARTING VALIDATION AND FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Load Validation Data
    val_df = load_dataset("val", load_cached_data=True, debug=Config.DEBUG)
    val_dataset = GNSSSequenceDataset(val_df, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_PATH}")
        return float("inf")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Inference Loop
    phone_errors = {}

    # storage for failure analysis
    feature_list = []
    error_list = []

    # Metadata list matches the order of the dataset
    seq_metadata = val_dataset.metadata
    seq_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            outputs = model(features)
            # Head 0 is the full resolution output (B, 2, L)
            preds = outputs[0]

            # Calculate Euclidean distance error per point: sqrt(dN^2 + dE^2)
            # Shape: (B, L)
            dists = torch.sqrt(torch.sum((preds - targets) ** 2, dim=1))

            # Move to CPU for processing
            dists_np = dists.cpu().numpy()
            mask_np = mask.cpu().numpy()
            feats_np = features.cpu().numpy()  # (B, C, L)

            batch_size = features.shape[0]

            for b in range(batch_size):
                drive_id, phone_name = seq_metadata[seq_idx]
                seq_idx += 1

                # Extract valid points based on mask
                valid_mask = mask_np[b] == 1
                valid_errors = dists_np[b][valid_mask]

                # Store errors per phone for metric calculation
                if phone_name not in phone_errors:
                    phone_errors[phone_name] = []
                phone_errors[phone_name].extend(valid_errors)

                # Store features and errors for failure analysis
                # Transpose features to (L, C) then select valid rows
                valid_features = feats_np[b].T[valid_mask]

                # Subsample large sequences to save memory if needed, but full analysis is better
                feature_list.append(valid_features)
                error_list.append(valid_errors)

    # 4. Compute Official Metric
    # Metric: Mean of (50th + 95th percentile) averaged across phones
    phone_scores = []
    print("\nPer-Phone Validation Scores:")
    for phone, errors in phone_errors.items():
        if len(errors) == 0:
            continue
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)
        print(
            f"  {phone}: {score:.4f} (p50: {p50:.4f}, p95: {p95:.4f}, count: {len(errors)})"
        )

    final_metric = np.mean(phone_scores) if phone_scores else float("inf")
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n" + "-" * 40)
    print("FAILURE ANALYSIS: Feature Correlations with Error")
    print("-" * 40)

    if feature_list:
        # Concatenate all data
        all_features = np.concatenate(feature_list, axis=0)
        all_errors = np.concatenate(error_list, axis=0)

        # Create DataFrame
        analysis_df = pd.DataFrame(all_features, columns=Config.FEATURE_COLS)
        analysis_df["Error"] = all_errors

        # Compute correlations
        correlations = (
            analysis_df.corr()["Error"].drop("Error").sort_values(ascending=False)
        )

        print("Correlation between Input Features and Prediction Error:")
        print(correlations)

        print("\nTop 3 Factors contributing to error:")
        print(correlations.head(3))
    else:
        print("No validation data available for analysis.")

    return final_metric


if __name__ == "__main__":
    # Set reproducibility
    set_seed(Config.SEED)

    # Optimize Config for Fast Baseline
    # Reduce epochs to ensure completion within time limit while maintaining learning
    Config.EPOCHS = 15

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Features: {len(Config.FEATURE_COLS)}")

    # 1. Train Model
    print("\n>>> Step 1: Training Model")
    train_model(load_cached_data=True)

    # 2. Validate and Analyze
    print("\n>>> Step 2: Validation and Analysis")
    metric = run_validation_and_analysis()

    # 3. Generate Submission
    # Threshold from instructions
    THRESHOLD = 3.802240262877392

    print("\n>>> Step 3: Submission Decision")
    if metric < THRESHOLD:
        print(f"Validation metric {metric} is better than threshold {THRESHOLD}.")
        print("Generating submission...")
        generate_predictions(load_cached_data=True)
    else:
        print(f"Validation metric {metric} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")
