import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import (
    SEED,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    DEVICE,
    FEATURE_DIM,
)
from library.utils import save_submission, seed_everything
from library.preprocessor import TabularTransformer
from library.dataset import get_dataloaders
from library.feature_extractor import CNNBackbone, process_and_cache_features
from library.classifier import MalignancyClassifier


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    seed_everything(SEED)
    demo_dir = os.path.join(WORKING_DIR, "demo")
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Step 1: Loading and subsetting metadata...")

    # Load original metadata
    df_train_full = pd.read_csv(TRAIN_CSV)
    df_val_full = pd.read_csv(VAL_CSV)
    df_test_full = pd.read_csv(TEST_CSV)

    # Subset for speed (50 train, 20 val, 20 test)
    # We use a small subset to ensure the script finishes quickly within the constraints.
    df_train = df_train_full.sample(n=50, random_state=SEED).reset_index(drop=True)
    df_val = df_val_full.sample(n=20, random_state=SEED).reset_index(drop=True)
    df_test = df_test_full.sample(n=20, random_state=SEED).reset_index(drop=True)

    print(f"  Train subset shape: {df_train.shape}")
    print(f"  Val subset shape: {df_val.shape}")
    print(f"  Test subset shape: {df_test.shape}")

    # 2. Tabular Preprocessing
    print("\nStep 2: Preprocessing tabular data...")

    # Instantiate the transformer
    tab_transformer = TabularTransformer()

    # Fit on training data
    tab_transformer.fit(df_train)

    # Transform all sets
    X_tab_train = tab_transformer.transform(df_train)
    X_tab_val = tab_transformer.transform(df_val)
    X_tab_test = tab_transformer.transform(df_test)

    # Verification
    assert X_tab_train.shape[0] == 50, "Train tabular rows mismatch"
    assert X_tab_val.shape[0] == 20, "Val tabular rows mismatch"
    assert X_tab_test.shape[0] == 20, "Test tabular rows mismatch"
    # Check that we have valid numerical data (no NaNs)
    assert not np.isnan(X_tab_train).any(), "NaNs found in transformed training data"

    print(f"  Tabular feature dimension: {X_tab_train.shape[1]}")

    # 3. Data Loaders
    print("\nStep 3: Creating DataLoaders...")

    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(
        df_train,
        df_val,
        df_test,
        X_tab_train,
        X_tab_val,
        X_tab_test,
        batch_size=batch_size,
        num_workers=2,
    )

    # Verify one batch
    images, tabs, targets = next(iter(train_loader))
    print(
        f"  Batch shapes - Images: {images.shape}, Tabular: {tabs.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Images: (B, 3, 224, 224)
    assert images.shape == (batch_size, 3, 224, 224), "Image batch shape incorrect"
    # Tabular: (B, n_features)
    assert tabs.shape == (
        batch_size,
        X_tab_train.shape[1],
    ), "Tabular batch shape incorrect"
    # Targets: (B,)
    assert targets.shape == (batch_size,), "Target batch shape incorrect"

    # 4. Feature Extraction
    print("\nStep 4: Extracting features using CNN Backbone...")

    # Initialize the backbone model once
    backbone = CNNBackbone()

    # Define cache paths for the demo
    train_feat_path = os.path.join(demo_dir, "train_feats.parquet")
    val_feat_path = os.path.join(demo_dir, "val_feats.parquet")
    test_feat_path = os.path.join(demo_dir, "test_feats.parquet")

    # Extract features
    # Note: load_cached_data=False forces extraction for demonstration
    print("  Extracting training features...")
    X_train_combined, y_train = process_and_cache_features(
        train_loader, train_feat_path, load_cached_data=False, model=backbone
    )

    print("  Extracting validation features...")
    X_val_combined, y_val = process_and_cache_features(
        val_loader, val_feat_path, load_cached_data=False, model=backbone
    )

    print("  Extracting test features...")
    X_test_combined, _ = process_and_cache_features(
        test_loader, test_feat_path, load_cached_data=False, model=backbone
    )

    # Verification
    # Expected dim = CNN feature dim (1280 for MobileNetV3 Large) + Tabular dim
    expected_dim = FEATURE_DIM + X_tab_train.shape[1]

    assert X_train_combined.shape == (
        50,
        expected_dim,
    ), f"Train feature shape mismatch: {X_train_combined.shape} vs expected (50, {expected_dim})"
    assert X_val_combined.shape == (
        20,
        expected_dim,
    ), f"Val feature shape mismatch: {X_val_combined.shape}"
    assert X_test_combined.shape == (
        20,
        expected_dim,
    ), f"Test feature shape mismatch: {X_test_combined.shape}"

    print(f"  Combined feature dimension: {expected_dim}")

    # 5. Classification
    print("\nStep 5: Training Classifier...")

    classifier = MalignancyClassifier()

    # Fit model
    classifier.fit(X_train_combined, y_train, X_val_combined, y_val)

    # Predict on validation to verify
    val_probs = classifier.predict_proba(X_val_combined)
    assert len(val_probs) == 20, "Validation predictions length mismatch"
    assert (val_probs >= 0).all() and (
        val_probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Save and Load verification
    model_path = "demo_model.joblib"
    classifier.save(model_path)

    loaded_clf = MalignancyClassifier().load(model_path)
    # Check if loaded model produces same predictions
    loaded_probs = loaded_clf.predict_proba(X_val_combined)
    np.testing.assert_array_almost_equal(
        val_probs, loaded_probs, err_msg="Model save/load inconsistency"
    )
    print("  Model persistence verified.")

    # 6. Submission Generation
    print("\nStep 6: Generating Submission...")

    # Predict on test set
    test_probs = classifier.predict_proba(X_test_combined)

    submission_path = os.path.join(demo_dir, "submission.csv")
    save_submission(df_test["image_name"].values, test_probs, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (20, 2), "Submission file shape incorrect"
    assert list(sub_df.columns) == [
        "image_name",
        "target",
    ], "Submission columns incorrect"
    assert sub_df["image_name"].equals(
        df_test["image_name"]
    ), "Image names in submission do not match test set"

    print("\n=== Demonstration Complete ===")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
