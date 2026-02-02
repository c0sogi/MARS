import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import pre-defined library modules
from library.config import Config
from library.trainer import Trainer
from library.preprocessing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset
from library.inference import generate_submission as generate_test_submission
from library.utils import seed_everything


def main():
    # =========================================================================
    # 1. Configuration and Setup
    # =========================================================================
    # Initialize configuration
    config = Config()

    # Modify config for a Fast Baseline execution
    config.NUM_EPOCHS = 1  # Train for only 1 epoch to ensure speed
    config.BATCH_SIZE = 32  # Batch size
    MAX_TRAIN_SAMPLES = 1000  # Limit training samples to 1000 for quick turnaround

    # Set seeds for reproducibility
    seed_everything(config.RANDOM_SEED)

    print(f"Initializing Fast Baseline Run...")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}, Max Train Samples: {MAX_TRAIN_SAMPLES}")

    # =========================================================================
    # 2. Model Training
    # =========================================================================
    # Initialize Trainer
    trainer = Trainer(config)

    # Train the model
    # This will preprocess data (if not cached), train the model, and save 'best_model.pth'
    model = trainer.train_model(load_cached_data=True, max_samples=MAX_TRAIN_SAMPLES)

    # =========================================================================
    # 3. Validation Assessment
    # =========================================================================
    print("\n--- Starting Validation Assessment ---")
    device = torch.device(config.DEVICE)
    model.eval()

    # Load Validation Data using Preprocessor
    preprocessor = GNSSPreprocessor()
    val_df = preprocessor.generate_dataset(split="val", load_cached_data=True)

    # Create Validation Dataset and Loader
    # We use the full validation set (no max_samples) to get the true metric
    val_dataset = GnssSequenceDataset(val_df, split="val", config=config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Batch size 1 for validation inference to handle variable sequence lengths
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    all_errors = []
    all_trip_ids = []
    feature_matrix = []

    print(f"Validating on {len(val_dataset)} sequences...")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            original_length = batch["original_length"].item()

            drive_id = batch["drive_id"][0]
            phone_name = batch["phone_name"][0]
            # Construct trip_id to group errors later
            trip_id = f"{drive_id}/{phone_name}"

            # Forward pass
            output = model(features)  # Shape: (1, OutputChannels, Length)

            # Unpad and transpose to (Length, Channels)
            pred_np = output[0, :, :original_length].cpu().numpy().transpose(1, 0)
            target_np = targets[0, :, :original_length].cpu().numpy().transpose(1, 0)
            feat_np = features[0, :, :original_length].cpu().numpy().transpose(1, 0)

            # Calculate Euclidean distance error (Meters)
            # Targets are DeltaNorth, DeltaEast
            diff = pred_np - target_np
            errors = np.sqrt(np.sum(diff**2, axis=1))

            all_errors.append(errors)
            all_trip_ids.extend([trip_id] * original_length)
            feature_matrix.append(feat_np)

    # Flatten collected data
    flat_errors = np.concatenate(all_errors)
    flat_trips = np.array(all_trip_ids)
    flat_features = np.concatenate(feature_matrix)

    # Compute Competition Metric
    # Mean of the (50th percentile + 95th percentile) / 2 for each phone
    unique_trips = np.unique(flat_trips)
    trip_scores = []

    for trip in unique_trips:
        trip_mask = flat_trips == trip
        trip_errs = flat_errors[trip_mask]

        p50 = np.percentile(trip_errs, 50)
        p95 = np.percentile(trip_errs, 95)
        score = (p50 + p95) / 2
        trip_scores.append(score)

    final_metric = np.mean(trip_scores)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")
    print("Correlation between Input Features and Error Magnitude:")

    feature_names = config.get_feature_names()
    correlations = []

    # Check if feature dimensions match
    if flat_features.shape[1] == len(feature_names):
        for i, name in enumerate(feature_names):
            # Calculate Pearson correlation
            # Handle constant features to avoid warnings
            if np.std(flat_features[:, i]) > 1e-9:
                corr = np.corrcoef(flat_features[:, i], flat_errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"{'Feature':<40} | {'Correlation':<10}")
        print("-" * 55)
        for name, corr in correlations[:10]:  # Print top 10
            print(f"{name:<40} | {corr:.4f}")
    else:
        print("Feature dimension mismatch, skipping detailed correlation analysis.")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 3.7864967500302016

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        # Use the inference library function to generate submission for the test set
        generate_test_submission(config=config, load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT better than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
