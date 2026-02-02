import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from unittest.mock import patch
import shutil

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import sparse_engine
from library import ranker_prep
from library import ranker_model
from library import image_processor


def run_demo():
    print("Starting H&M Recommendation System Demo...")

    # 1. Setup and Configuration
    utils.seed_everything(42)

    # Override config for speed
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_rounds"] = 5
    config.TOP_K_RETRIEVAL = 20  # Reduce candidates for demo

    # Clean working directory to ensure fresh run
    if config.WORKING_DIR.exists():
        shutil.rmtree(config.WORKING_DIR)
    config.WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Define Mocking Strategy for Speed
    # We wrap the original data loaders to return subsets
    print("\n[Demo] Patching data loaders to use data subsets...")

    original_load_transactions = data_loader.load_transactions
    original_load_articles = data_loader.load_articles
    original_load_customers = data_loader.load_customers
    original_load_sample_sub = data_loader.load_sample_submission

    def mocked_load_transactions(split="train", **kwargs):
        # Load full but slice immediately
        # Using 200k rows to ensure enough overlap for graph
        print(f"  -> Mock loading {split} transactions (subset)")
        df = original_load_transactions(split=split, debug_nrows=200000, **kwargs)
        return df

    def mocked_load_articles(**kwargs):
        print("  -> Mock loading articles (subset)")
        # Load enough articles to cover the transaction subset
        df = original_load_articles(debug_nrows=5000, **kwargs)
        return df

    def mocked_load_customers(**kwargs):
        print("  -> Mock loading customers (subset)")
        df = original_load_customers(debug_nrows=10000, **kwargs)
        return df

    def mocked_load_sample_sub(**kwargs):
        print("  -> Mock loading sample submission (subset)")
        df = original_load_sample_sub(debug_nrows=1000, **kwargs)
        return df

    # Apply patches
    with patch(
        "library.data_loader.load_transactions", side_effect=mocked_load_transactions
    ), patch(
        "library.data_loader.load_articles", side_effect=mocked_load_articles
    ), patch(
        "library.data_loader.load_customers", side_effect=mocked_load_customers
    ), patch(
        "library.data_loader.load_sample_submission", side_effect=mocked_load_sample_sub
    ):

        # ==========================================
        # STAGE 1: SPARSE RETRIEVAL
        # ==========================================
        print("\n=== Stage 1: Sparse Retrieval Engine ===")

        # Load data manually to demonstrate fit
        train_df = data_loader.load_transactions(split="train")

        retriever = sparse_engine.SparseGraphRetriever()
        retriever.fit(train_df, load_cached_data=False)

        # Validation
        assert (
            retriever.transition_matrix is not None
        ), "Transition matrix should be built"
        assert (
            retriever.transition_matrix.nnz > 0
        ), "Transition matrix should not be empty"
        print(f"Transition Matrix Shape: {retriever.transition_matrix.shape}")

        # Test Prediction on a few users
        test_users = train_df["customer_id"].unique()[:5]
        candidates = retriever.predict(test_users)

        assert not candidates.empty, "Candidates dataframe should not be empty"
        assert "score" in candidates.columns
        assert "rank" in candidates.columns
        print(f"Generated {len(candidates)} candidates for {len(test_users)} users.")

        # ==========================================
        # STAGE 2: RANKER DATASET PREPARATION
        # ==========================================
        print("\n=== Stage 2: Ranker Dataset Preparation ===")

        dataset_builder = ranker_prep.RankerDatasetBuilder()

        # Build Train Set
        # This will trigger:
        # 1. Load subset transactions
        # 2. Fit retriever on history (subset)
        # 3. Generate candidates for target (subset)
        # 4. Compute image embeddings (subset of articles)
        # 5. Merge metadata
        print("Building Train Set...")
        train_set = dataset_builder.build_ranker_train_set(load_cached_data=False)

        assert not train_set.empty, "Train set should not be empty"
        assert "label" in train_set.columns, "Train set must have label"
        assert "visual_similarity" in train_set.columns, "Visual features missing"
        print(f"Train Set Shape: {train_set.shape}")
        print(f"Positive Labels: {train_set['label'].sum()}")

        # Build Validation Set
        print("Building Validation Set...")
        val_set = dataset_builder.build_ranker_val_set(load_cached_data=False)

        assert not val_set.empty, "Validation set should not be empty"
        print(f"Validation Set Shape: {val_set.shape}")

        # ==========================================
        # STAGE 3: RANKER MODEL TRAINING
        # ==========================================
        print("\n=== Stage 3: LightGBM Ranker Training ===")

        ranker = ranker_model.LGBMRankerWrapper()

        # Fit model
        ranker.fit(train_set, val_set)

        assert ranker.model is not None, "Model should be trained"

        # Check Feature Importance
        importance = ranker.get_feature_importance()
        print("\nTop 5 Features:")
        print(importance.head(5))

        assert not importance.empty, "Feature importance should be available"

        # ==========================================
        # STAGE 4: INFERENCE & SUBMISSION
        # ==========================================
        print("\n=== Stage 4: Inference and Submission ===")

        # Build Inference Set (Test Users)
        print("Building Inference Set...")
        test_set = dataset_builder.build_inference_set(load_cached_data=False)

        assert not test_set.empty, "Test set should not be empty"

        # Generate Submission
        submission_path = config.SUBMISSION_DIR / "submission.csv"
        ranker.generate_submission(test_set, submission_path)

        # Verify Submission
        assert submission_path.exists(), "Submission file was not created"

        sub_df = pd.read_csv(submission_path)
        print(f"Submission generated with {len(sub_df)} rows.")
        print("Head of submission:")
        print(sub_df.head())

        # Check format
        assert "customer_id" in sub_df.columns
        assert "prediction" in sub_df.columns
        # Check prediction format (space separated)
        sample_pred = sub_df.iloc[0]["prediction"]
        if isinstance(sample_pred, str) and len(sample_pred) > 0:
            items = sample_pred.split()
            assert len(items) <= 12, "Should not predict more than 12 items"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
