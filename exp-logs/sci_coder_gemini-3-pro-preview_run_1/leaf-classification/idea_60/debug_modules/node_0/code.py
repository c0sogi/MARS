import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library import config, utils, features, data, model


def demo_utils():
    print("\n=== Demo: Utils Module ===")

    # 1. Test Seed Setting
    utils.set_seed(42)
    print("Seed set to 42.")

    # 2. Test Metadata Loading
    # We expect these files to exist based on the prompt description
    try:
        df_train = utils.load_metadata("train")
        df_val = utils.load_metadata("val")
        df_test = utils.load_metadata("test")

        print(f"Metadata loaded successfully.")
        print(f"Train shape: {df_train.shape}")
        print(f"Val shape:   {df_val.shape}")
        print(f"Test shape:  {df_test.shape}")

        # Assertions
        assert not df_train.empty, "Train metadata is empty"
        assert "species" in df_train.columns, "Train metadata missing 'species' column"
        assert (
            "file_path" in df_train.columns
        ), "Train metadata missing 'file_path' column"

    except FileNotFoundError as e:
        print(f"Metadata file missing: {e}")
        raise e

    # 3. Test Timer
    with utils.Timer("Test Timer Block"):
        x = sum([i**2 for i in range(1000)])

    print("Utils demo completed successfully.")


def demo_features():
    print("\n=== Demo: Features Module ===")

    # Use a small limit to ensure speed
    debug_limit = 10

    print(f"Running process_dataset for 'train' split with limit={debug_limit}...")

    # Force computation from scratch to test logic (load_cached_data=False)
    # Note: The library saves cache even in debug mode if not careful,
    # but process_dataset logic says: if debug_limit: df = df.head...
    # and "if not debug_limit: utils.save_cache..." so it won't overwrite main cache.
    df_features = features.process_dataset(
        "train", load_cached_data=False, debug_limit=debug_limit
    )

    print(f"Processed DataFrame shape: {df_features.shape}")

    # Check if new geometric features are present
    expected_geo_features = config.GEOMETRIC_FEATURES
    for feat in expected_geo_features:
        assert (
            feat in df_features.columns
        ), f"Geometric feature {feat} missing from output"

    # Check values
    # Area should be > 0 for valid images
    if "area" in df_features.columns:
        areas = df_features["area"].values
        print(f"Sample Areas: {areas[:5]}")
        # We can't strictly assert area > 0 because some images might be empty/corrupt,
        # but in this dataset they should be valid.
        assert np.all(areas >= 0), "Negative area detected"

    print("Features demo completed successfully.")


def demo_data_manager():
    print("\n=== Demo: Data Manager Module ===")

    dm = data.LeafDataManager()

    # Use a slightly larger limit to ensure variance calculation doesn't fail
    # (need >1 sample, ideally enough to have variance)
    debug_limit = 50

    print(f"Preparing data with debug_limit={debug_limit}...")

    # We use load_cached_data=False to verify the pipeline logic
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dm.prepare_data(
        load_cached_data=False, debug_limit=debug_limit
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Classes: {classes}")

    # Assertions
    assert len(X_train) == len(y_train), "Mismatch in X_train and y_train length"
    assert X_train.shape[1] > 0, "No features remaining after sanitization"
    assert not np.isnan(X_train).any(), "NaNs found in X_train"
    assert not np.isinf(X_train).any(), "Infs found in X_train"

    # Check scaling (StandardScaler should make mean ~0 and std ~1)
    # With small debug_limit, this might fluctuate, but let's check roughly
    mean_val = np.mean(X_train, axis=0)
    std_val = np.std(X_train, axis=0)

    print(f"Mean of first 5 features: {mean_val[:5]}")
    print(f"Std of first 5 features: {std_val[:5]}")

    # Allow some tolerance due to float precision and small N
    assert np.all(np.abs(mean_val) < 1e-5), "Data not centered (Mean != 0)"

    print("Data Manager demo completed successfully.")
    return X_train, y_train, X_val, y_val, classes


def demo_model(X_train, y_train, X_val, y_val, classes):
    print("\n=== Demo: Model Module ===")

    print("Initializing SanitizedOASDiscriminant...")
    clf = model.SanitizedOASDiscriminant()

    print("Fitting model...")
    clf.fit(X_train, y_train)

    print("Predicting probabilities on validation set...")
    probs = clf.predict_proba(X_val)

    print(f"Probabilities shape: {probs.shape}")

    # Assertions
    assert probs.shape == (len(X_val), len(classes)), "Probability shape mismatch"

    # Check sum to 1
    row_sums = np.sum(probs, axis=1)
    # Floating point tolerance
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check range
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"

    # Check prediction labels
    preds = clf.predict(X_val)
    assert len(preds) == len(X_val), "Prediction length mismatch"

    # Calculate Log Loss (just to ensure it runs)
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    loss = log_loss(y_val, probs_clipped)
    print(f"Validation Log Loss (Debug Subset): {loss:.4f}")

    print("Model demo completed successfully.")


def demo_full_pipeline():
    print("\n=== Demo: Full Pipeline Execution ===")

    # Run the high-level function provided in model.py
    # This orchestrates everything and generates a submission
    debug_limit = 30

    print(f"Running run_training_pipeline with debug_limit={debug_limit}...")
    model.run_training_pipeline(load_cached_data=False, debug_limit=debug_limit)

    # Verify submission file creation
    submission_path = config.SUBMISSION_FILE_PATH
    if os.path.exists(submission_path):
        print(f"Submission file found at: {submission_path}")
        df_sub = pd.read_csv(submission_path)
        print(f"Submission shape: {df_sub.shape}")

        # Verify columns
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        # Check if we have probability columns (should match number of classes in training)
        # In debug mode, we might have fewer classes if the subset doesn't contain all of them,
        # but the code should handle it gracefully or the subset is stratified.
        # The provided metadata split is stratified, but taking head(30) might break that.
        # However, the code should still produce a valid CSV structure.

        # Check values
        prob_cols = [c for c in df_sub.columns if c != "id"]
        probs = df_sub[prob_cols].values
        assert np.all(probs >= 0) and np.all(
            probs <= 1
        ), "Submission probabilities out of range"
    else:
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    print("Full pipeline demo completed successfully.")


if __name__ == "__main__":
    try:
        # 1. Utils
        demo_utils()

        # 2. Features
        demo_features()

        # 3. Data Manager
        # We pass the data from here to the model demo to save time re-computing
        X_train, y_train, X_val, y_val, classes = demo_data_manager()

        # 4. Model
        demo_model(X_train, y_train, X_val, y_val, classes)

        # 5. Full Pipeline
        demo_full_pipeline()

        print("\nAll demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
