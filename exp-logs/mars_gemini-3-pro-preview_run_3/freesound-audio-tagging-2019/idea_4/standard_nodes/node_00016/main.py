import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_lwlrap
from library.dataset import AudioDataset
from library.model import AudioClassifier
from library.engine import fit_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets and DataLoaders...")
    # Load cached data is True to utilize any pre-processed metadata
    train_dataset = AudioDataset(split="train", load_cached_data=True)
    val_dataset = AudioDataset(split="val", load_cached_data=True)

    # Use Config parameters for batch size and workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("Initializing AudioClassifier...")
    model = AudioClassifier()

    # 4. Training Loop
    # fit_model handles the entire training process including early stopping
    # and loading the best model weights at the end.
    print("Starting Training...")
    model = fit_model(model, train_loader, val_loader)

    # 5. Final Validation Assessment
    print("Performing final validation inference...")
    model.eval()

    val_targets_list = []
    val_preds_list = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            # Collect results (move to CPU for numpy conversion)
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    # Concatenate all batches
    val_preds = np.vstack(val_preds_list)
    val_targets = np.vstack(val_targets_list)

    # Compute Metric
    final_metric = calculate_lwlrap(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error Magnitude: Mean Absolute Error per sample
    # Shape: (n_samples, n_classes)
    error_matrix = np.abs(val_preds - val_targets)
    # Average error across all classes for each sample
    mean_error_per_sample = np.mean(error_matrix, axis=1)

    # Input Feature: Polyphony (Number of ground truth labels)
    # We sum the binary target vector to get the count of labels
    polyphony = np.sum(val_targets, axis=1)

    # Calculate Pearson Correlation
    # We use numpy's corrcoef which returns a matrix [[1, r], [r, 1]]
    if np.std(mean_error_per_sample) == 0 or np.std(polyphony) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(mean_error_per_sample, polyphony)[0, 1]

    print(
        f"Correlation between Error Magnitude and Polyphony (Num Labels): {correlation:.10f}"
    )

    # 7. Submission Generation
    THRESHOLD = 0.7117108825122853

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = AudioDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds_list = []
        test_fnames_list = []

        # Inference Loop
        with torch.no_grad():
            for inputs, fnames in test_loader:
                inputs = inputs.to(device)

                logits = model(inputs)
                preds = torch.sigmoid(logits)

                test_preds_list.append(preds.cpu().numpy())
                test_fnames_list.extend(fnames)

        test_preds = np.vstack(test_preds_list)

        # Format Submission
        # We need the class names in the correct order.
        # The dataset and sample_submission ensure this consistency.
        sample_sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
        class_columns = [c for c in sample_sub_df.columns if c != "fname"]

        # Create DataFrame
        submission_df = pd.DataFrame(test_preds, columns=class_columns)
        submission_df.insert(0, "fname", test_fnames_list)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
