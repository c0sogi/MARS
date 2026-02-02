import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import provided library modules
from library.config import Config, StreamConfig
from library.utils import set_seed, check_files_exist, calculate_metric
from library.data_loader import DogDataset, get_class_mapping
from library.feature_extractor import get_concatenated_features
from library.model_trainer import (
    train_logistic_head,
    predict_stream,
    optimize_ensemble_weights,
    generate_submission,
)


def setup_demo_environment():
    """
    Creates a lightweight subset of the data to ensure the demo runs quickly.
    Modifies the global Config to point to this subset.
    """
    print("Setting up demo environment...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Select top 3 breeds to ensure we have enough samples for Cross-Validation
    # even with a small subset.
    top_breeds = train_df["breed"].value_counts().head(3).index.tolist()
    print(f"Selected breeds for demo: {top_breeds}")

    # Filter Train: Take 10 samples per breed (30 total)
    demo_train = (
        train_df[train_df["breed"].isin(top_breeds)]
        .groupby("breed")
        .head(10)
        .reset_index(drop=True)
    )

    # Filter Val: Take 5 samples per breed (15 total)
    demo_val = (
        val_df[val_df["breed"].isin(top_breeds)]
        .groupby("breed")
        .head(5)
        .reset_index(drop=True)
    )

    # Filter Test: Take first 10 samples (arbitrary)
    demo_test = test_df.head(10).copy()

    # Define paths for demo metadata
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    # Save demo metadata
    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Monkey-patch Config to use these new paths
    # This redirects all library functions to use our small dataset
    Config.TRAIN_METADATA = demo_train_path
    Config.VAL_METADATA = demo_val_path
    Config.TEST_METADATA = demo_test_path

    # Update working directory for outputs to keep them separate
    Config.WORKING_DIR = os.path.join(Config.WORKING_DIR, "demo_outputs")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    return len(demo_train), len(demo_val), len(demo_test)


def verify_data_loading():
    """
    Verifies that the DogDataset and transforms are working correctly.
    """
    print("\n--- Verifying Data Loader ---")

    # Initialize dataset with Stream A config
    dataset = DogDataset(Config.TRAIN_METADATA, Config.STREAM_A, mode="train")

    # Fetch one item
    item = dataset[0]

    # Check keys
    required_keys = ["id", "global", "standard", "local", "label"]
    for key in required_keys:
        assert key in item, f"Missing key {key} in dataset item"

    # Check tensor shapes for Stream A (Input size 224)
    # Shape should be (3, 224, 224)
    expected_shape = (3, 224, 224)
    assert item["global"].shape == expected_shape
    assert item["standard"].shape == expected_shape
    assert item["local"].shape == expected_shape

    print("Data Loader verification passed: Shapes and keys are correct.")


def run_pipeline():
    # 1. Initialization
    set_seed(42)
    n_train, n_val, n_test = setup_demo_environment()

    # 2. Verify Data Loading
    verify_data_loading()

    # --- STREAM A: ConvNeXt-Large ---
    print("\n" + "=" * 40)
    print("Processing Stream A (ConvNeXt-Large)")
    print("=" * 40)

    # Feature Extraction
    # We disable caching (load_cached_data=False) to force the extraction logic to run
    print("Extracting features...")
    X_train_a, ids_train_a, y_train_a = get_concatenated_features(
        Config.STREAM_A, mode="train", load_cached_data=False
    )
    X_val_a, ids_val_a, y_val_a = get_concatenated_features(
        Config.STREAM_A, mode="val", load_cached_data=False
    )
    X_test_a, ids_test_a, _ = get_concatenated_features(
        Config.STREAM_A, mode="test", load_cached_data=False
    )

    # Validation: Check feature dimensions
    # ConvNeXt-Large embedding dim is 1536. 3 views -> 1536 * 3 = 4608
    expected_dim_a = 1536 * 3
    assert X_train_a.shape == (
        n_train,
        expected_dim_a,
    ), f"Stream A Train shape mismatch: {X_train_a.shape}"
    assert X_val_a.shape == (
        n_val,
        expected_dim_a,
    ), f"Stream A Val shape mismatch: {X_val_a.shape}"
    print(f"Stream A features extracted successfully. Shape: {X_train_a.shape}")

    # Model Training
    print("Training Logistic Regression Head for Stream A...")
    # Using low max_iter for speed in demo
    model_a = train_logistic_head(
        X_train_a, y_train_a, "stream_a", load_cached_model=False, max_iter=100
    )

    # Prediction
    probs_val_a = predict_stream(model_a, X_val_a)
    probs_test_a = predict_stream(model_a, X_test_a)

    score_a = calculate_metric(y_val_a, probs_val_a)
    print(f"Stream A Validation Log Loss: {score_a:.4f}")

    # --- STREAM B: RegNetY-128GF ---
    print("\n" + "=" * 40)
    print("Processing Stream B (RegNetY-128GF)")
    print("=" * 40)

    # Feature Extraction
    print("Extracting features...")
    X_train_b, ids_train_b, y_train_b = get_concatenated_features(
        Config.STREAM_B, mode="train", load_cached_data=False
    )
    X_val_b, ids_val_b, y_val_b = get_concatenated_features(
        Config.STREAM_B, mode="val", load_cached_data=False
    )
    X_test_b, ids_test_b, _ = get_concatenated_features(
        Config.STREAM_B, mode="test", load_cached_data=False
    )

    # Validation: Check feature dimensions
    # RegNetY-128GF embedding dim is 7392. 3 views -> 7392 * 3 = 22176
    expected_dim_b = 7392 * 3
    assert X_train_b.shape == (
        n_train,
        expected_dim_b,
    ), f"Stream B Train shape mismatch: {X_train_b.shape}"
    print(f"Stream B features extracted successfully. Shape: {X_train_b.shape}")

    # Model Training
    print("Training Logistic Regression Head for Stream B...")
    model_b = train_logistic_head(
        X_train_b, y_train_b, "stream_b", load_cached_model=False, max_iter=100
    )

    # Prediction
    probs_val_b = predict_stream(model_b, X_val_b)
    probs_test_b = predict_stream(model_b, X_test_b)

    score_b = calculate_metric(y_val_b, probs_val_b)
    print(f"Stream B Validation Log Loss: {score_b:.4f}")

    # --- ENSEMBLING ---
    print("\n" + "=" * 40)
    print("Ensemble Optimization")
    print("=" * 40)

    # Verify alignment of validation sets
    assert np.array_equal(
        y_val_a, y_val_b
    ), "Validation labels mismatch between streams"

    # Find optimal weights
    weights = optimize_ensemble_weights(probs_val_a, probs_val_b, y_val_a)

    # --- SUBMISSION ---
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    # Weighted average of test probabilities
    final_probs = weights[0] * probs_test_a + weights[1] * probs_test_b

    # Renormalize (just to be safe)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

    # Generate CSV
    submission_file = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    generate_submission(ids_test_a, final_probs, output_path=submission_file)

    # Verify output
    assert os.path.exists(submission_file), "Submission file was not created"

    # Check content of submission
    sub_df = pd.read_csv(submission_file)
    print(f"Submission generated with shape: {sub_df.shape}")
    print("First 3 rows:")
    print(sub_df.head(3))

    # Since we filtered to 3 breeds, the submission should have 4 columns (id + 3 breeds)
    assert (
        sub_df.shape[1] == 4
    ), f"Expected 4 columns (id + 3 breeds), got {sub_df.shape[1]}"

    print("\nDemo Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_pipeline()
