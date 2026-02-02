import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.image_extractor import DeepFeatureExtractor
from library.data_pipeline import DataPipeline
from library.modeling import (
    get_discriminative_solver,
    get_generative_solver,
    train_predict,
)


def run_demo():
    print("Starting Library Usage Demonstration...")

    # =========================================================================
    # 1. Configuration Overrides for Speed
    # =========================================================================
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    # Reduce the hyperparameter search grid for Logistic Regression
    Config.LR_CS = np.logspace(-1, 1, 3)
    # Reduce Cross-Validation folds
    Config.CV_FOLDS = 2
    # Reduce max iterations for solver
    Config.LR_MAX_ITER = 100
    # Reduce batch size for inference
    Config.BATCH_SIZE = 16

    # Set a global seed for this script
    np.random.seed(Config.RANDOM_SEED)

    print("Configuration updated: Small Grid Search, 2-Fold CV, Max Iter 100.")

    # =========================================================================
    # 2. Demonstrate Deep Feature Extractor
    # =========================================================================
    print("\n[Step 2] Testing DeepFeatureExtractor component...")

    # Initialize the extractor
    extractor = DeepFeatureExtractor()

    # Load a few sample paths from the training metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {train_meta_path}")

    df_train_sample = pd.read_csv(train_meta_path).head(5)

    # Construct full paths (library expects list of strings)
    sample_paths = [
        os.path.join(Config.INPUT_DIR, p) for p in df_train_sample["image_path"]
    ]

    # Run extraction (force no cache to verify computation)
    print(f"Extracting features for {len(sample_paths)} sample images...")
    features = extractor.extract(sample_paths, load_cached_data=False)

    # Verification
    expected_shape = (5, 512)  # ResNet18 output before FC is 512
    assert (
        features.shape == expected_shape
    ), f"Extractor output shape mismatch. Expected {expected_shape}, got {features.shape}"
    assert not np.isnan(features).any(), "Extracted features contain NaNs."

    print("DeepFeatureExtractor verification successful.")

    # =========================================================================
    # 3. Demonstrate Data Pipeline
    # =========================================================================
    print("\n[Step 3] Running full DataPipeline...")

    pipeline = DataPipeline()

    # Execute the pipeline
    # This loads metadata, merges train/val, extracts handcrafted features,
    # extracts deep features (cached or fresh), and applies PCA/Scaling.
    data_dict = pipeline.run(load_cached_data=True)

    # Inspect the output dictionary
    required_keys = [
        "X_train_view1",
        "X_train_view2",
        "y_train",
        "X_test_view1",
        "X_test_view2",
        "test_ids",
        "classes",
    ]

    for key in required_keys:
        assert key in data_dict, f"Pipeline output missing key: {key}"

    # Extract data for modeling
    X_train_v1 = data_dict["X_train_view1"]  # Handcrafted
    X_train_v2 = data_dict["X_train_view2"]  # Deep Features (PCA)
    y_train = data_dict["y_train"]
    X_test_v1 = data_dict["X_test_view1"]
    X_test_v2 = data_dict["X_test_view2"]
    classes = data_dict["classes"]

    n_samples_train = len(y_train)
    n_samples_test = len(data_dict["test_ids"])

    # Verification
    print(f"Processed Train Samples: {n_samples_train}")
    print(f"Processed Test Samples: {n_samples_test}")
    print(f"View 1 (Handcrafted) Features: {X_train_v1.shape[1]}")
    print(f"View 2 (Deep PCA) Features: {X_train_v2.shape[1]}")

    assert X_train_v1.shape[0] == n_samples_train
    assert X_train_v2.shape[0] == n_samples_train
    assert X_test_v1.shape[0] == n_samples_test
    assert X_test_v2.shape[0] == n_samples_test
    assert X_train_v1.shape[1] == 192, "View 1 should have 192 handcrafted features."

    print("DataPipeline verification successful.")

    # =========================================================================
    # 4. Demonstrate Modeling (Discriminative & Generative)
    # =========================================================================
    print("\n[Step 4] Training and Predicting with Library Models...")

    # A. Discriminative Solver (Logistic Regression CV)
    # We use View 1 (Handcrafted) for this demonstration
    print("--- Model A: Discriminative Solver (LR) on View 1 ---")
    lr_model = get_discriminative_solver()

    # Train and Predict
    probs_lr = train_predict(
        lr_model, X_train_v1, y_train, X_test_v1, model_name="LogisticRegression_View1"
    )

    # Verify LR Output
    assert probs_lr.shape == (
        n_samples_test,
        len(classes),
    ), f"LR prediction shape mismatch. Got {probs_lr.shape}"
    # Check probability sum
    row_sums = probs_lr.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "LR probabilities do not sum to 1."

    # B. Generative Solver (LDA)
    # We use View 2 (Deep Features) for this demonstration
    print("\n--- Model B: Generative Solver (LDA) on View 2 ---")
    lda_model = get_generative_solver()

    # Train and Predict
    probs_lda = train_predict(
        lda_model, X_train_v2, y_train, X_test_v2, model_name="LDA_View2"
    )

    # Verify LDA Output
    assert probs_lda.shape == (
        n_samples_test,
        len(classes),
    ), f"LDA prediction shape mismatch. Got {probs_lda.shape}"
    assert np.allclose(
        probs_lda.sum(axis=1), 1.0, atol=1e-5
    ), "LDA probabilities do not sum to 1."

    print("Modeling verification successful.")

    # =========================================================================
    # 5. Final Output Check
    # =========================================================================
    print("\n[Step 5] Creating Sample Submission DataFrame...")

    # Create a dataframe to mimic submission format
    submission_df = pd.DataFrame(probs_lr, columns=classes)
    submission_df.insert(0, "id", data_dict["test_ids"])

    print("Sample Submission Head:")
    print(submission_df.head(3).to_string())

    assert submission_df.shape == (
        99,
        100,
    ), f"Submission shape mismatch. Expected (99, 100), got {submission_df.shape}"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
