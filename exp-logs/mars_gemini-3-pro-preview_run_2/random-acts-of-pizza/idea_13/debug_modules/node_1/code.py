import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import BaggingClassifier

# Import library modules
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import PizzaDataLoader
from library.feature_engineering import TextEmbedder, TabularProcessor, FeatureFuser
from library.model_factory import ModelFactory
from library.training_pipeline import TrainingPipeline


def main():
    # 1. Setup and Configuration Override for Speed
    print(">>> Setting up environment and overriding config for demonstration...")
    setup_logger(level=logging.INFO)
    set_seed(42)

    # Patch Config to run a lightweight version of the pipeline
    # Original: 5 splits, 10 estimators, extensive grid search
    # Demo: 2 splits, 2 estimators, minimal grid search
    Config.N_SPLITS = 2
    Config.BAGGING_N_ESTIMATORS = 2
    Config.GRID_SEARCH_PARAMS = {
        "C": [0.1, 1.0],  # Reduced to 2 options
        "alpha": [1.0],  # Fixed alpha
        "class_weight": [None],  # Fixed class weight
    }

    # Ensure working directory is clean for specific demo files if needed
    # (The pipeline handles its own caching, which is fine)

    # 2. Demonstrate Data Loading
    print("\n>>> Demonstrating PizzaDataLoader...")
    data_loader = PizzaDataLoader()
    # Force reload to demonstrate processing logic, though caching is handled inside
    train_df, val_df, test_df = data_loader.load_data(load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Assertions
    assert not train_df.empty, "Train DataFrame is empty"
    assert "requester_received_pizza" in train_df.columns, "Target missing in train"
    assert "request_id" in test_df.columns, "ID missing in test"

    # 3. Demonstrate Feature Engineering
    print("\n>>> Demonstrating TextEmbedder...")
    text_embedder = TextEmbedder()

    # We'll use a small subset for the standalone embedder test to be instant
    # The pipeline will use the full set later
    subset_df = train_df.head(10).copy()
    embeddings = text_embedder.get_embeddings(
        subset_df, "train_demo_subset", load_cached_data=False
    )

    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (10, 384), f"Expected (10, 384), got {embeddings.shape}"
    # Check L2 normalization (norm should be approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not L2 normalized"

    print("\n>>> Demonstrating TabularProcessor...")
    tabular_processor = TabularProcessor()
    # Process full datasets
    X_train_tab, X_val_tab, X_test_tab = tabular_processor.process_numeric_features(
        train_df, val_df, test_df
    )

    print(f"Tabular Train shape: {X_train_tab.shape}")
    assert X_train_tab.shape[0] == len(train_df), "Row count mismatch in tabular train"
    assert X_train_tab.shape[1] == len(
        Config.NUMERIC_FEATURES
    ), "Feature count mismatch"
    # Check if RankGauss worked (output should be roughly normal, mean ~0 std ~1)
    # We check a single feature
    feat_mean = np.mean(X_train_tab[:, 0])
    feat_std = np.std(X_train_tab[:, 0])
    print(f"Feature 0 Stats -> Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")
    assert abs(feat_mean) < 0.5, "RankGauss mean not close to 0"

    print("\n>>> Demonstrating FeatureFuser...")
    # Create dummy embeddings matching the full train size for fusion test
    dummy_embeddings = np.random.rand(len(train_df), 384)
    dummy_embeddings = dummy_embeddings / np.linalg.norm(
        dummy_embeddings, axis=1, keepdims=True
    )

    alpha = 2.0
    fused_features = FeatureFuser.fuse(dummy_embeddings, X_train_tab, alpha)
    print(f"Fused features shape: {fused_features.shape}")

    expected_dim = 384 + len(Config.NUMERIC_FEATURES)
    assert fused_features.shape == (
        len(train_df),
        expected_dim,
    ), "Fusion dimension mismatch"

    # Verify scaling logic: Tabular part should be scaled by alpha
    # Tabular features are at the end
    tabular_part = fused_features[:, 384:]
    assert np.allclose(tabular_part, X_train_tab * alpha), "Differential scaling failed"

    # 4. Demonstrate Model Factory
    print("\n>>> Demonstrating ModelFactory...")
    model = ModelFactory.create_bagged_ensemble(C=0.1, n_estimators=5)
    assert isinstance(
        model, BaggingClassifier
    ), "Factory did not return a BaggingClassifier"
    print("Model created successfully.")

    # 5. Run Full Pipeline (Integration Test)
    print("\n>>> Running Full TrainingPipeline (with reduced config)...")
    pipeline = TrainingPipeline()
    pipeline.run()

    # 6. Verify Submission
    print("\n>>> Verifying Submission...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    assert sub_df.shape[0] == len(test_df), "Submission row count mismatch"
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect"
    assert (
        sub_df["requester_received_pizza"].between(0, 1).all()
    ), "Probabilities out of bounds"

    print("\n>>> Demonstration and Verification Complete!")


if __name__ == "__main__":
    main()
