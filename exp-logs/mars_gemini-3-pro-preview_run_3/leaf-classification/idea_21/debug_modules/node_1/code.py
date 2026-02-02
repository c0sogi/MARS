import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import logging

# Import Library modules
from library.config import Config
from library.utils import seed_everything
from library.feature_extraction import FeatureExtractor
from library.data_processing import OrthogonalDataManager
from library.pipeline import create_expert_pipeline
from library.training import OSLDETrainer
from library.inference import predict_test_set


def main():
    print("=== OS-LDE Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1/6] Configuring environment...")
    seed_everything(42)

    # Define a specific working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    DATA_DIR = os.path.join(DEMO_DIR, "data")
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Override Config parameters for speed and demo isolation
    # We modify the class attributes directly before other classes use them
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.N_FOLDS = 2  # Minimum folds for cross-validation
    Config.DEBUG = (
        False  # We manage data subsetting manually to ensure stratification safety
    )

    # ---------------------------------------------------------
    # 2. Data Subsetting
    # ---------------------------------------------------------
    print("[2/6] Preparing data subsets...")

    # Load original metadata
    full_train_df = pd.read_csv("./metadata/train.csv")
    full_test_df = pd.read_csv("./metadata/test.csv")

    # Create a balanced training subset
    # We need enough samples per class to survive StratifiedKFold with n_splits=2
    # We pick the top 3 most frequent species and take 4 samples from each.
    top_species = full_train_df["species"].value_counts().head(3).index.tolist()
    train_subset = (
        full_train_df[full_train_df["species"].isin(top_species)]
        .groupby("species")
        .head(4)
    )

    # Create a small test subset
    test_subset = full_test_df.head(6)

    # Save subsets to the demo data directory
    train_subset_path = os.path.join(DATA_DIR, "train_subset.csv")
    test_subset_path = os.path.join(DATA_DIR, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # Update Config to point to these new metadata files
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.TEST_METADATA_PATH = test_subset_path

    print(
        f"   Training subset: {len(train_subset)} samples, {len(top_species)} classes."
    )
    print(f"   Test subset: {len(test_subset)} samples.")

    # ---------------------------------------------------------
    # 3. Feature Extraction
    # ---------------------------------------------------------
    print("\n[3/6] Executing Feature Extraction...")
    extractor = FeatureExtractor()

    # A. Extract Train Features
    print("   Processing Training Images...")
    train_data = extractor.extract_features(
        dataset_type="train", load_cached_data=False
    )

    # Verify Train Outputs
    # Expected Shape: (N_samples, 12_views, 2560_features)
    expected_train_shape = (len(train_subset), 12, 2560)
    assert (
        train_data["features"].shape == expected_train_shape
    ), f"Train features shape mismatch. Expected {expected_train_shape}, got {train_data['features'].shape}"
    assert train_data["tabular"].shape == (
        len(train_subset),
        192,
    ), "Train tabular shape mismatch"
    assert len(train_data["labels"]) == len(
        train_subset
    ), "Train labels length mismatch"

    # B. Extract Test Features
    print("   Processing Test Images...")
    test_data = extractor.extract_features(dataset_type="test", load_cached_data=False)

    # Verify Test Outputs
    expected_test_shape = (len(test_subset), 12, 2560)
    assert (
        test_data["features"].shape == expected_test_shape
    ), f"Test features shape mismatch. Expected {expected_test_shape}, got {test_data['features'].shape}"

    print("   Feature extraction verified.")

    # ---------------------------------------------------------
    # 4. Data Processing (Orthogonal Partitioning)
    # ---------------------------------------------------------
    print("\n[4/6] Partitioning Data into Orthogonal Sets...")
    data_manager = OrthogonalDataManager()

    # Process Train Data
    # This splits the 12 views into sets A, B, and C and computes centroids
    processed_train = data_manager.get_data("train", load_cached_data=False)
    centroids_train = processed_train["centroids"]

    # Verify Centroids
    for key in ["A", "B", "C"]:
        assert key in centroids_train, f"Missing centroid set {key}"
        assert centroids_train[key].shape == (
            len(train_subset),
            2560,
        ), f"Centroid {key} shape mismatch"

    # Process Test Data
    processed_test = data_manager.get_data("test", load_cached_data=False)
    centroids_test = processed_test["centroids"]
    assert centroids_test["A"].shape == (
        len(test_subset),
        2560,
    ), "Test centroid shape mismatch"

    print("   Orthogonal partitioning verified.")

    # ---------------------------------------------------------
    # 5. Pipeline Training
    # ---------------------------------------------------------
    print("\n[5/6] Training Experts...")
    trainer = OSLDETrainer()

    # Execute training loop
    # This trains (N_FOLDS * 3_experts) models
    avg_log_loss = trainer.train_and_evaluate()

    print(f"   Training completed. Average Log Loss: {avg_log_loss:.6f}")

    # Verify Model Artifacts
    # We expect files like: model_fold_0_expert_A.pkl
    for fold in range(Config.N_FOLDS):
        for expert in ["A", "B", "C"]:
            model_name = f"model_fold_{fold}_expert_{expert}.pkl"
            model_path = os.path.join(DEMO_DIR, model_name)
            assert os.path.exists(model_path), f"Model artifact missing: {model_name}"

            # Verify we can load it and it's a pipeline
            model = joblib.load(model_path)
            assert hasattr(
                model, "predict_proba"
            ), f"Artifact {model_name} is not a valid classifier"

    print("   All model artifacts verified.")

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6/6] Running Inference...")

    # Generate predictions
    predict_test_set(load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    # Columns = id + number of classes
    # Rows = number of test samples
    label_encoder = joblib.load(os.path.join(DEMO_DIR, "label_encoder.pkl"))
    num_classes = len(label_encoder.classes_)

    assert len(submission_df) == len(test_subset), "Submission row count mismatch"
    assert (
        len(submission_df.columns) == num_classes + 1
    ), "Submission column count mismatch"
    assert "id" in submission_df.columns, "ID column missing"

    # Check probability constraints
    # Sum of probabilities per row should be ~1.0
    prob_cols = [c for c in submission_df.columns if c != "id"]
    row_sums = submission_df[prob_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    # Check ID matching
    assert np.all(
        submission_df["id"].values == test_subset["id"].values
    ), "Test IDs do not match input"

    print(f"   Submission file verified at: {Config.SUBMISSION_PATH}")
    print("\n=== Demo Execution Successful ===")


if __name__ == "__main__":
    main()
