import os
import sys
import numpy as np
import pandas as pd
import torch
import random

# Import provided library modules
from library.config import Config
from library.data_loader import load_essay_data
from library.feature_extractor import EmbeddingEngine
from library.model_trainer import RidgeRegressor
from library.utils import post_process_preds


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup and Configuration Override
    print("=== Setting up Environment ===")
    set_seeds(Config.SEED)
    Config.setup()

    # Override Config for speed in this demonstration
    print("Overriding Config for demonstration speed...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small sample for quick execution
    Config.NUM_WORKERS = 1

    # 2. Data Loading
    print("\n=== Data Loading ===")
    # Load Train Data
    train_ids, train_texts, train_scores = load_essay_data("train")
    assert (
        len(train_ids) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples, got {len(train_ids)}"
    assert len(train_texts) == len(train_ids), "Mismatch between train IDs and texts"
    assert len(train_scores) == len(train_ids), "Mismatch between train IDs and scores"

    # Load Validation Data
    val_ids, val_texts, val_scores = load_essay_data("val")
    assert len(val_ids) > 0, "Validation set is empty"

    # Load Test Data
    test_ids, test_texts, test_scores = load_essay_data("test")
    assert len(test_ids) > 0, "Test set is empty"
    assert test_scores is None, "Test scores should be None"

    print("Data loaded successfully.")

    # 3. Feature Extraction (Embeddings)
    print("\n=== Feature Extraction ===")
    engine = EmbeddingEngine()

    # Generate embeddings for Train
    print("Encoding training set...")
    X_train = engine.generate_embeddings(train_texts, "train")

    # Validate embedding shape
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    expected_dim = 384
    assert X_train.shape == (
        len(train_texts),
        expected_dim,
    ), f"Expected shape {(len(train_texts), expected_dim)}, got {X_train.shape}"

    # Generate embeddings for Validation
    print("Encoding validation set...")
    X_val = engine.generate_embeddings(val_texts, "val")
    assert X_val.shape == (len(val_texts), expected_dim)

    # Generate embeddings for Test
    print("Encoding test set...")
    X_test = engine.generate_embeddings(test_texts, "test")
    assert X_test.shape == (len(test_texts), expected_dim)

    print("Feature extraction complete.")

    # 4. Model Training
    print("\n=== Model Training ===")
    regressor = RidgeRegressor()

    # Train the model
    # We pass validation data to see evaluation metrics during training
    regressor.train(X_train, train_scores, X_val, val_scores)

    # Validate that model weights were learned
    assert regressor.coef_ is not None, "Model coefficients are None after training"
    assert regressor.intercept_ is not None, "Model intercept is None after training"
    assert regressor.coef_.shape[0] == expected_dim, "Coefficient shape mismatch"

    # Check if model file was saved
    assert os.path.exists(
        regressor.model_path
    ), f"Model file not found at {regressor.model_path}"

    print("Model training and validation complete.")

    # 5. Inference and Submission
    print("\n=== Inference and Submission ===")

    # Predict on Test Set
    raw_preds = regressor.predict(X_test)
    assert len(raw_preds) == len(test_ids), "Prediction count mismatch"

    # Post-process predictions (clip, round, cast to int)
    final_preds = post_process_preds(raw_preds)

    # Validate predictions
    assert np.issubdtype(final_preds.dtype, np.integer), "Predictions must be integers"
    assert (
        final_preds.min() >= 1 and final_preds.max() <= 6
    ), "Predictions out of range [1, 6]"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"essay_id": test_ids, "score": final_preds})

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Final Validation of Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Read back to verify format
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_check.columns) == [
        "essay_id",
        "score",
    ], "Incorrect columns in submission file"
    assert len(df_check) == len(test_ids), "Row count mismatch in submission file"

    print("\n=== Pipeline Execution Successful ===")
    print(f"Submission generated with {len(df_check)} rows.")
    print(df_check.head())


if __name__ == "__main__":
    main()
