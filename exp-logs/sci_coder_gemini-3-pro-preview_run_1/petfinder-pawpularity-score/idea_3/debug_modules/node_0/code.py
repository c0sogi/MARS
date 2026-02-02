import sys
import os
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.utils import set_seed, IMG_SIZE
from library.data import create_dataloaders, PetDataset
from library.feature_extractor import DualBackboneExtractor
from library.model import RidgeRegressor


def main():
    # 1. Setup and Configuration
    print("Step 1: Setup and Configuration")
    set_seed(42)

    # Define debug parameters for speed
    DEBUG_SAMPLE_SIZE = 32
    BATCH_SIZE = 8

    print(f"Debug Sample Size: {DEBUG_SAMPLE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")

    # 2. Data Loading Demonstration
    print("\nStep 2: Data Loading Demonstration")
    # We use create_dataloaders with debug=True to get small subsets
    train_loader, val_loader, test_loader = create_dataloaders(
        debug=True, debug_sample_size=DEBUG_SAMPLE_SIZE, batch_size=BATCH_SIZE
    )

    # Verify Train Loader
    # Expecting (Batch_Size, Channels, H, W) for images
    # Expecting (Batch_Size, 12) for metadata
    # Expecting (Batch_Size,) for targets
    try:
        images, meta, targets = next(iter(train_loader))
        print(
            f"Train Batch - Images: {images.shape}, Meta: {meta.shape}, Targets: {targets.shape}"
        )

        assert images.shape == (
            BATCH_SIZE,
            3,
            IMG_SIZE,
            IMG_SIZE,
        ), f"Expected image shape {(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)}, got {images.shape}"
        assert meta.shape == (
            BATCH_SIZE,
            12,
        ), f"Expected meta shape {(BATCH_SIZE, 12)}, got {meta.shape}"
        assert targets.shape == (
            BATCH_SIZE,
        ), f"Expected targets shape {(BATCH_SIZE,)}, got {targets.shape}"

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader (returns IDs instead of targets)
    try:
        t_images, t_meta, t_ids = next(iter(test_loader))
        print(
            f"Test Batch - Images: {t_images.shape}, Meta: {t_meta.shape}, IDs Batch Size: {len(t_ids)}"
        )

        assert (
            len(t_ids) == BATCH_SIZE
        ), f"Expected test batch size {BATCH_SIZE}, got {len(t_ids)}"

    except StopIteration:
        raise AssertionError("Test loader is empty!")

    # 3. Feature Extraction Demonstration
    print("\nStep 3: Feature Extraction Demonstration")
    # Initialize the extractor (loads Swin and ConvNeXt models)
    extractor = DualBackboneExtractor()

    # We manually call _extract_from_loader on our debug loaders to avoid
    # processing the full dataset which get_features() would do by default.

    print("Extracting features from debug train loader...")
    train_feats, train_meta_extracted, train_targets_extracted = (
        extractor._extract_from_loader(train_loader, is_test=False)
    )

    print(f"Extracted Train Features Shape: {train_feats.shape}")
    print(f"Extracted Train Meta Shape: {train_meta_extracted.shape}")

    # Validation: Check dimensions
    # Swin Large (1536) + ConvNeXt Large (1536) = 3072
    EXPECTED_FEAT_DIM = 3072
    assert (
        train_feats.shape[1] == EXPECTED_FEAT_DIM
    ), f"Expected feature dimension {EXPECTED_FEAT_DIM}, got {train_feats.shape[1]}"
    # Check sample count: drop_last=True is used in train_loader
    expected_train_samples = len(train_loader.dataset) - (
        len(train_loader.dataset) % BATCH_SIZE
    )
    assert (
        train_feats.shape[0] == expected_train_samples
    ), f"Feature count {train_feats.shape[0]} does not match expected batch-aligned count {expected_train_samples}"

    print("Extracting features from debug test loader (subset)...")
    # Note: Test loader in debug mode (via create_dataloaders) might still be full size
    # depending on implementation, but it's small enough (992 samples) to run quickly.
    test_feats, test_meta_extracted, test_ids_extracted = (
        extractor._extract_from_loader(test_loader, is_test=True)
    )
    print(f"Extracted Test Features Shape: {test_feats.shape}")

    # 4. Model Training Demonstration
    print("\nStep 4: Model Training Demonstration")

    # Prepare Training Data: Concatenate Image Features + Metadata
    X_train = np.hstack([train_feats, train_meta_extracted])
    y_train = train_targets_extracted

    print(f"Final Training Input Shape: {X_train.shape}")

    # Initialize RidgeRegressor
    # Using cv=2 to ensure it works with small sample sizes
    model = RidgeRegressor(cv=2)

    # Fit Model
    print("Fitting RidgeRegressor...")
    model.fit(X_train, y_train)

    # Predict on Training Set (Sanity Check)
    train_preds = model.predict(X_train)
    train_rmse = model.get_rmse(y_train, train_preds)
    print(f"Training RMSE (on debug subset): {train_rmse:.4f}")

    assert not np.isnan(train_preds).any(), "Predictions contain NaNs"
    assert train_rmse >= 0, "RMSE should be non-negative"

    # 5. Submission Generation Demonstration
    print("\nStep 5: Submission Generation Demonstration")

    # Prepare Test Data
    X_test = np.hstack([test_feats, test_meta_extracted])

    # Predict
    test_preds = model.predict(X_test)

    # Create Submission DataFrame
    submission = pd.DataFrame({"Id": test_ids_extracted, "Pawpularity": test_preds})

    print("Generated Submission DataFrame Head:")
    print(submission.head())

    # Validate Submission Format
    assert "Id" in submission.columns, "Submission missing 'Id' column"
    assert (
        "Pawpularity" in submission.columns
    ), "Submission missing 'Pawpularity' column"
    assert len(submission) == len(test_ids_extracted), "Submission row count mismatch"

    # Check value range (Pawpularity is 1-100), though regression might output slightly outside
    print(
        f"Prediction Range: {submission['Pawpularity'].min():.2f} - {submission['Pawpularity'].max():.2f}"
    )

    print("\nSuccess: All demonstrations and validations completed.")


if __name__ == "__main__":
    main()
