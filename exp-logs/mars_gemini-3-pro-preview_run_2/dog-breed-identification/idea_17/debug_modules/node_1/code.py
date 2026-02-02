import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config
from library import data_loader
from library import feature_engine
from library import hybrid_model


def run_demo():
    print("Initializing Demo Run...")

    # ==========================================
    # 1. Runtime Configuration Patching
    # ==========================================
    # Patch config to use a temporary directory and faster settings for the demo
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    print(f"Patching configuration for speed...")
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    config.NUM_WORKERS = 0  # Avoid overhead for small debug batch

    # Reduce complexity for Logistic Regression
    config.LOGREG_PARAMS["cv"] = 2
    config.LOGREG_PARAMS["max_iter"] = 10

    # Reduce complexity for Gradient Boosting
    config.GB_PARAMS["max_iter"] = 10
    config.GB_PARAMS["n_iter_no_change"] = 2
    config.GB_PARAMS["validation_fraction"] = 0.2

    # Set seeds for reproducibility
    feature_engine.set_seed(config.SEED)

    # ==========================================
    # 2. Data Loading (Debug Mode)
    # ==========================================
    print("\n--- Step 1: Loading Data (Debug Mode) ---")
    # debug=True limits the dataset to 100 samples per split
    train_loader, val_loader, test_loader, class_to_idx = data_loader.get_dataloaders(
        debug=True
    )

    # Verification: Check batch structure
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")

    # Assertions
    assert "global" in batch, "Missing 'global' view in batch"
    assert "standard" in batch, "Missing 'standard' view in batch"
    assert "local" in batch, "Missing 'local' view in batch"
    assert "label" in batch, "Missing 'label' in training batch"

    # Check shape: (Batch, 3, 224, 224)
    img_shape = batch["global"].shape
    print(f"Image tensor shape: {img_shape}")
    assert len(img_shape) == 4, "Image tensor should be 4D"
    assert img_shape[1] == 3, "Image should have 3 channels"
    assert img_shape[2] == 224 and img_shape[3] == 224, "Image size should be 224x224"

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\n--- Step 2: Extracting Features ---")
    # We force load_cached_data=False to ensure we actually test the extraction logic
    # (though in this new dir, cache wouldn't exist anyway)

    # Train Features
    X_train, y_train, ids_train = feature_engine.extract_features(
        train_loader, "train", load_cached_data=False
    )

    # Val Features
    X_val, y_val, ids_val = feature_engine.extract_features(
        val_loader, "val", load_cached_data=False
    )

    # Test Features
    X_test, _, ids_test = feature_engine.extract_features(
        test_loader, "test", load_cached_data=False
    )

    print(f"Extracted Train Features Shape: {X_train.shape}")

    # Assertions
    expected_dim = 1536 * 3  # ConvNeXt Large (1536) * 3 Views
    assert (
        X_train.shape[1] == expected_dim
    ), f"Expected feature dim {expected_dim}, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch between features and labels count"
    assert len(X_train) <= 100, "Debug mode should limit samples to <= 100"

    # ==========================================
    # 4. Hybrid Model Training
    # ==========================================
    print("\n--- Step 3: Training Hybrid Ensemble ---")
    model = hybrid_model.HybridEnsemble()

    # Fit the model
    model.fit(X_train, y_train)

    # Verify classes
    print(f"Number of classes learned: {len(model.classes_)}")
    assert len(model.classes_) > 0, "Model failed to learn any classes"

    # Optimize weights
    print("Optimizing weights...")
    val_loss = model.optimize_weights(X_val, y_val)

    print(f"Optimized Linear Weight: {model.w_linear:.4f}")

    # Assertions
    assert 0.0 <= model.w_linear <= 1.0, "Ensemble weight out of bounds"

    # Save model
    model_save_path = os.path.join(demo_working_dir, "model")
    model.save(model_save_path)
    assert os.path.exists(
        os.path.join(model_save_path, "ensemble_meta.joblib")
    ), "Model save failed"

    # ==========================================
    # 5. Prediction & Submission
    # ==========================================
    print("\n--- Step 4: Generating Submission ---")
    submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    hybrid_model.generate_submission(model, X_test, ids_test, submission_path)

    # Validation
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(f"First few columns: {sub_df.columns[:5].tolist()}")

    # Check structure
    # 1 column for ID + 120 columns for breeds
    expected_cols = 121
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"
    assert "id" in sub_df.columns, "Missing 'id' column"
    assert sub_df.shape[0] == len(ids_test), "Submission row count mismatch"

    # Check probability validity
    # Sum of probabilities (excluding ID) should be approx 1.0
    probs = sub_df.drop(columns=["id"]).values
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("\nDemo Run Completed Successfully!")


if __name__ == "__main__":
    run_demo()
