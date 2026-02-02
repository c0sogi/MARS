import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import library modules
import library.config as config
import library.utils as utils
import library.image_processing as ip
import library.data_loader as dl
import library.preprocessing as pp
import library.model as md


def run_demo():
    print("=== Starting Demonstration of Leaf Classification Pipeline ===\n")

    # 1. Setup and Seeding
    print("1. Setting random seeds for reproducibility...")
    utils.set_seed(config.SEED)

    # Clean up working directory to ensure fresh run for demonstration
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    print("   Working directory cleaned.\n")

    # 2. Image Processing Demonstration
    print("2. Testing Image Processing Module...")

    # Load metadata to find a valid image path
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    sample_img_path = os.path.join(
        config.IMAGES_BASE_DIR, sample_row[config.FILE_PATH_COL]
    )

    print(f"   Extracting features from sample image: {sample_img_path}")
    geo_features = ip.extract_geometric_features(sample_img_path)

    # Validation: Check feature keys and types
    expected_keys = set(config.GEOMETRIC_FEATURES)
    assert set(geo_features.keys()) == expected_keys, "Geometric feature keys mismatch."
    assert all(
        isinstance(v, float) for v in geo_features.values()
    ), "Features must be floats."
    print(f"   Extracted {len(geo_features)} geometric features successfully.")
    print(
        f"   Sample values: Area={geo_features['Area']:.2f}, Solidity={geo_features['Solidity']:.4f}"
    )

    # Validation: Process Dataset (Subset)
    print("   Processing a subset of the training dataset (limit=10)...")
    # We use a small limit to speed up the demo
    # Note: process_dataset doesn't natively support limit, but data_loader does.
    # We will test data_loader directly in the next step which calls process_dataset.
    print(
        "   (Skipping direct process_dataset call to avoid redundant computation, testing via data_loader)\n"
    )

    # 3. Data Loading Demonstration
    print("3. Testing Data Loader Module...")
    limit_size = 50
    print(f"   Loading datasets with limit={limit_size} and load_cached_data=False...")

    train_df, val_df, test_df = dl.load_datasets(
        load_cached_data=False, limit=limit_size
    )

    # Validation: Check shapes
    assert (
        len(train_df) == limit_size
    ), f"Train DF length mismatch. Expected {limit_size}, got {len(train_df)}"
    assert (
        len(val_df) == limit_size
    ), f"Val DF length mismatch. Expected {limit_size}, got {len(val_df)}"
    # Test DF might be smaller if the file is smaller than limit, but here limit is small
    assert len(test_df) <= limit_size

    # Validation: Check columns
    # Should have ID, Target (for train/val), Tabular Features, and Geometric Features
    expected_cols = set(config.ALL_FEATURES + [config.ID_COL, config.FILE_PATH_COL])
    # Train/Val should have species
    assert config.TARGET_COL in train_df.columns
    assert set(config.ALL_FEATURES).issubset(
        train_df.columns
    ), "Missing feature columns in Train DF."

    print(f"   Successfully loaded {len(train_df)} training samples.")
    print(f"   Feature columns count: {len(config.ALL_FEATURES)}")
    print("   Data Loader verification passed.\n")

    # 4. Preprocessing Demonstration
    print("4. Testing Preprocessing Module...")

    # A. Test SanitizedTransformer logic independently
    print("   A. Verifying SanitizedTransformer logic...")
    transformer = pp.SanitizedTransformer()

    # Create synthetic data: 10 samples, 5 features
    # Feature 0: Constant (should be removed)
    # Feature 1: Random
    X_synth = np.random.rand(10, 5)
    X_synth[:, 0] = 1.0

    transformer.fit(X_synth)
    X_trans = transformer.transform(X_synth)

    # Expect 4 features remaining (1 constant removed)
    assert (
        X_trans.shape[1] == 4
    ), f"VarianceThreshold failed. Expected 4 features, got {X_trans.shape[1]}"
    print("      SanitizedTransformer correctly removed constant feature.")

    # B. Run full preprocessing on loaded data
    print("   B. Running full preprocessing pipeline on loaded datasets...")
    X_train, y_train, X_val, y_val, X_test, classes = pp.get_preprocessed_data(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation
    assert X_train.shape[0] == limit_size
    assert X_train.dtype == config.FLOAT_PRECISION
    assert len(classes) > 0
    print(
        f"   Preprocessing complete. X_train shape: {X_train.shape}, Classes: {len(classes)}"
    )
    print("   Preprocessing verification passed.\n")

    # 5. Model Demonstration
    print("5. Testing Model Module (OASDiscriminant)...")

    model = md.OASDiscriminant(assume_centered=True)
    print("   Fitting model...")
    model.fit(X_train, y_train)

    print("   Predicting probabilities on validation set...")
    val_probs = model.predict_proba(X_val)

    # Validation: Output shape
    assert val_probs.shape == (len(X_val), len(classes)), "Prediction shape mismatch."

    # Validation: Probabilities sum to 1
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    # Validation: Check Log Loss calculation
    print("   Calculating Log Loss...")
    loss = utils.compute_log_loss(y_val, val_probs, classes=np.arange(len(classes)))
    print(f"   Validation Log Loss: {loss:.4f}")
    assert loss > 0, "Log loss should be positive."
    print("   Model verification passed.\n")

    # 6. Full Pipeline Execution
    print("6. Testing Full Pipeline (train_and_predict)...")

    # We use a slightly larger limit to ensure we cover enough classes for OAS to be stable,
    # although OAS is robust to n_features > n_samples.
    pipeline_limit = 100

    # Force reload to ensure pipeline runs end-to-end
    print(f"   Running pipeline with limit={pipeline_limit}...")
    md.train_and_predict(load_cached_data=False, limit=pipeline_limit)

    # Validation: Check submission file
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"   Submission file created at {config.SUBMISSION_PATH}")
        print(f"   Submission shape: {sub_df.shape}")

        # Reload classes from the cache to ensure we validate against the correct set
        # generated by the pipeline run with limit=100. Cite debug_lesson_6.
        classes_path = os.path.join(config.WORKING_DIR, "classes.npy")
        if os.path.exists(classes_path):
            classes = np.load(classes_path, allow_pickle=True)
            print(f"   Reloaded {len(classes)} classes from pipeline cache.")

        # Check columns: id + class names
        assert config.ID_COL in sub_df.columns
        assert len(sub_df.columns) == len(classes) + 1  # +1 for id

        # Check values are within [0, 1] (though clipping handles this, just sanity check)
        prob_cols = [c for c in sub_df.columns if c != config.ID_COL]
        probs = sub_df[prob_cols].values
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), "Submission contains invalid probabilities."

        print("   Full pipeline verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
