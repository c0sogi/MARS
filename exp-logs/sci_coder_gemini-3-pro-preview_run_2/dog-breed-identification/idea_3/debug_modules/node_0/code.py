import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import MODELS, SUBMISSION_PATH, SAMPLE_SUBMISSION_CSV, DEVICE
from library.utils import seed_everything
from library.data import create_dataloaders
from library.feature_extractor import run_feature_extraction
from library.classifier import run_classification_pipeline


def validate_dataloader(model_name):
    """
    Validates that the DataLoader yields batches with correct shapes and types.
    """
    print(f"\n--- Validating DataLoader for {model_name} ---")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        model_name, batch_size=4
    )

    # Check class count
    assert len(classes) == 120, f"Expected 120 classes, found {len(classes)}"
    print(f"Number of classes: {len(classes)}")

    # Fetch one batch from train loader
    images, labels, ids = next(iter(train_loader))

    # Verify Image Tensor
    # Expected: (Batch, Channels, Height, Width)
    print(f"Image batch shape: {images.shape}")
    assert images.dim() == 4, "Images should be a 4D tensor (B, C, H, W)"
    assert images.shape[0] == 4, "Batch size mismatch"
    assert images.shape[1] == 3, "Images should have 3 channels (RGB)"

    # Verify Labels
    print(f"Labels shape: {labels.shape}")
    assert labels.dim() == 1, "Labels should be a 1D tensor"
    assert labels.shape[0] == 4, "Label batch size mismatch"
    assert isinstance(labels, torch.Tensor), "Labels should be a torch Tensor"

    # Verify IDs
    assert len(ids) == 4, "ID list length mismatch"
    assert isinstance(ids[0], str), "IDs should be strings"

    print("DataLoader validation passed.")


def validate_submission():
    """
    Validates the generated submission file against the sample submission requirements.
    """
    print("\n--- Validating Submission File ---")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    sub_df = pd.read_csv(SUBMISSION_PATH)
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    print(f"Submission shape: {sub_df.shape}")

    # Check dimensions
    # 1023 test images (based on metadata analysis) + header
    # 120 breeds + 1 id column = 121 columns
    expected_rows = len(sample_df)
    expected_cols = 121

    assert (
        sub_df.shape[0] == expected_rows
    ), f"Expected {expected_rows} rows, got {sub_df.shape[0]}"
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"

    # Check ID consistency
    # Sort both by ID to ensure matching order for comparison
    sub_ids = sorted(sub_df["id"].tolist())
    sample_ids = sorted(sample_df["id"].tolist())
    assert sub_ids == sample_ids, "Submission IDs do not match sample submission IDs"

    # Check Probabilities
    # Drop ID column to check numeric values
    probs = sub_df.drop(columns=["id"]).values

    # Check for NaN or Inf
    assert not np.isnan(probs).any(), "Submission contains NaN values"
    assert not np.isinf(probs).any(), "Submission contains Inf values"

    # Check if probabilities sum to approximately 1
    row_sums = probs.sum(axis=1)
    # Allow small floating point error
    assert np.allclose(
        row_sums, 1.0, atol=1e-4
    ), f"Probabilities do not sum to 1. Max deviation: {np.abs(row_sums - 1.0).max()}"

    print("Submission file validation passed.")


def main():
    # 1. Setup
    print("Initializing Script...")
    seed_everything(42)

    # 2. Data Loading Validation
    # We validate using one model configuration to ensure data pipeline is healthy
    validate_dataloader("convnext_large")

    # 3. Feature Extraction
    # We iterate over all models defined in config.MODELS
    # This step uses the GPU to extract features and saves them to cache.
    # If cache exists, it loads from disk.
    all_features = {}

    print("\n--- Starting Feature Extraction Phase ---")
    print(f"Device: {DEVICE}")

    for model_name in MODELS.keys():
        print(f"\nProcessing Model: {model_name}")

        # Run extraction (handles caching internally)
        # We set load_cached_data=True to use existing files if available (saves time on re-runs)
        features = run_feature_extraction(model_name, load_cached_data=True)

        # Basic validation of extracted features
        train_emb, train_lbl, _ = features["train"]
        print(f"  Train Embeddings: {train_emb.shape}")

        expected_dim = MODELS[model_name]["embedding_dim"]
        assert (
            train_emb.shape[1] == expected_dim
        ), f"Expected embedding dim {expected_dim}, got {train_emb.shape[1]}"

        all_features[model_name] = features

    # 4. Classification & Ensembling
    print("\n--- Starting Classification Phase ---")
    # This trains Logistic Regression on top of features and generates submission.csv
    # We set load_cached_models=False to demonstrate the training process explicitly.
    run_classification_pipeline(all_features, load_cached_models=False)

    # 5. Final Validation
    validate_submission()

    print("\nScript execution completed successfully.")


if __name__ == "__main__":
    main()
