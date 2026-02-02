import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
import library.feature_extractor as feature_extractor
import library.classifier as classifier
import library.ensemble as ensemble


def main():
    # 1. Setup and Reproducibility
    print("Setting up demonstration...")
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Create Lightweight Datasets for Speed
    # We must ensure the training set contains all 120 breeds to maintain
    # the correct output dimensionality for the classifier.
    print("Creating subset metadata for rapid execution...")

    orig_train = pd.read_csv(config.TRAIN_CSV)
    orig_val = pd.read_csv(config.VAL_CSV)
    orig_test = pd.read_csv(config.TEST_CSV)

    # Stratified sample: 1 image per breed for training (120 samples total)
    demo_train = orig_train.groupby("breed", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    )

    # Small random samples for validation and test
    demo_val = orig_val.sample(n=20, random_state=42)
    demo_test = orig_test.sample(n=20, random_state=42)

    # Save demo metadata files
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(config.WORKING_DIR, "demo_test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(
        f"Subsets created: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
    )

    # 3. Monkey-Patch Configuration
    # Redirect config to use our demo datasets
    config.TRAIN_CSV = demo_train_path
    config.VAL_CSV = demo_val_path
    config.TEST_CSV = demo_test_path

    # Reduce computational cost for the demo
    config.ENSEMBLE["max_iter"] = 100  # Fewer iterations for Logistic Regression
    config.ENSEMBLE["cv_folds"] = 2  # Fewer CV folds

    # We set this to False to force the code to run the extraction/training logic
    # instead of looking for non-existent cache files.
    FORCE_RUN = False

    # 4. Pipeline Execution: Stream A (ConvNeXt-Large)
    print("\n" + "=" * 40)
    print("Running Stream A Pipeline (ConvNeXt)")
    print("=" * 40)

    # A. Train the Classifier Head
    # This step implicitly handles:
    # 1. Feature Extraction (Global, Standard, Local views) for Train split
    # 2. Feature Fusion
    # 3. Logistic Regression Training
    clf_a = classifier.train_stream_head(config.STREAM_A, load_cached_data=FORCE_RUN)

    # B. Generate Predictions
    # This triggers feature extraction/fusion for Val and Test splits
    print("Generating Stream A predictions...")
    probs_a_val, ids_a_val, y_a_val = classifier.predict_stream(
        config.STREAM_A, split="val", load_cached_data=FORCE_RUN
    )
    probs_a_test, ids_a_test, _ = classifier.predict_stream(
        config.STREAM_A, split="test", load_cached_data=FORCE_RUN
    )

    # Validation check
    assert probs_a_val.shape == (20, 120), "Stream A Val output shape incorrect"

    # 5. Pipeline Execution: Stream B (RegNetY-128GF)
    print("\n" + "=" * 40)
    print("Running Stream B Pipeline (RegNetY)")
    print("=" * 40)

    # A. Train Head
    clf_b = classifier.train_stream_head(config.STREAM_B, load_cached_data=FORCE_RUN)

    # B. Generate Predictions
    print("Generating Stream B predictions...")
    probs_b_val, ids_b_val, y_b_val = classifier.predict_stream(
        config.STREAM_B, split="val", load_cached_data=FORCE_RUN
    )
    probs_b_test, ids_b_test, _ = classifier.predict_stream(
        config.STREAM_B, split="test", load_cached_data=FORCE_RUN
    )

    # Validation check
    assert probs_b_val.shape == (20, 120), "Stream B Val output shape incorrect"

    # 6. Ensemble Optimization
    print("\n" + "=" * 40)
    print("Optimizing Ensemble")
    print("=" * 40)

    # Calculate optimal weights based on validation log loss
    # We can use load_cached_data=True here because we just generated the files above.
    weights = ensemble.optimize_ensemble_weights(load_cached_data=True)

    # 7. Submission Generation
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    ensemble.generate_submission(weights, load_cached_data=True)

    # 8. Final Verification
    print("\n" + "=" * 40)
    print("Verifying Output")
    print("=" * 40)

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file missing at {config.SUBMISSION_PATH}")

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Check dimensions: 20 test samples + header, 121 columns (id + 120 breeds)
    expected_shape = (20, 121)
    if sub_df.shape != expected_shape:
        raise AssertionError(
            f"Submission shape {sub_df.shape} != expected {expected_shape}"
        )

    # Check ID alignment
    # The first column should be 'id'
    if sub_df.columns[0] != "id":
        raise AssertionError("First column of submission is not 'id'")

    # Check that IDs match our test subset
    submission_ids = set(sub_df["id"].values)
    test_subset_ids = set(demo_test["id"].values)

    if submission_ids != test_subset_ids:
        raise AssertionError("Submission IDs do not match test subset IDs")

    print("Success! All pipeline steps completed and verified.")


if __name__ == "__main__":
    main()
