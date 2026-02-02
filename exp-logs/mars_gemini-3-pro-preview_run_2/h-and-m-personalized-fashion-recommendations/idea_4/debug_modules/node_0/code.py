import pandas as pd
import numpy as np
import os
import sys
import shutil
import random
import torch
from pathlib import Path

# Import from the provided library
from library.config import Config
from library.data_factory import DataFactory
from library.retrieval import CandidateEngine
from library.features import FeatureEngineer
from library.ranker import LGBMRanker


# -------------------------------------------------------------------------
# 1. Setup & Configuration Overrides
# -------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_for_speed():
    """
    Overrides Config parameters to ensure the demo runs quickly.
    """
    print("Configuring parameters for speed...")
    # Reduce Embedding complexity
    Config.EMBED_DIM = 8
    Config.W2V_EPOCHS = 1
    Config.W2V_WINDOW = 3
    Config.W2V_NEGATIVE = 2

    # Reduce Retrieval counts
    Config.TOP_K_COOC = 5
    Config.TOP_K_EMBED = 5
    Config.TOP_K_REPURCHASE = 5
    Config.TOP_K_POPULARITY = 5

    # Reduce LightGBM complexity
    Config.LGBM_NUM_BOOST_ROUND = 10
    Config.LGBM_EARLY_STOPPING_ROUNDS = 5
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["learning_rate"] = 0.1

    # Ensure working directory is clean for this run
    if Config.WORKING_DIR.exists():
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()


# -------------------------------------------------------------------------
# 2. Main Execution Flow
# -------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(Config.SEED)
    configure_for_speed()

    print("\n=== Step 1: Data Loading & Sampling ===")
    # Load full data
    full_df = DataFactory.load_full_data(load_cached_data=False)

    # SAMPLING: Reduce to a tiny subset of users for demonstration speed
    # We pick 500 random users who have at least some history
    unique_users = full_df["customer_id"].unique()
    sampled_users = np.random.choice(
        unique_users, size=min(500, len(unique_users)), replace=False
    )

    # Filter transactions to these users
    sampled_df = full_df[full_df["customer_id"].isin(sampled_users)].copy()
    print(
        f"Sampled dataset: {len(sampled_df)} transactions from {len(sampled_users)} users."
    )

    # Split into Train History and Validation Ground Truth
    # Note: We disable caching here to ensure we use our sampled split
    train_history, val_ground_truth = DataFactory.get_time_split(
        sampled_df, load_cached_data=False
    )

    # Validation
    assert not train_history.empty, "Train history is empty"
    assert (
        train_history["t_dat"].max() <= val_ground_truth["t_dat"].min()
    ), "Data leakage detected in time split"
    print("Data split verified.")

    print("\n=== Step 2: Candidate Retrieval ===")
    # Initialize Engine
    engine = CandidateEngine()

    # Fit the retrieval models (Embeddings, Co-occurrence, Popularity)
    # This will save artifacts to ./working
    engine.fit(train_history)

    # Generate candidates for the validation users
    # We treat validation users as the 'test' users for whom we need recommendations
    val_users = pd.DataFrame({"customer_id": val_ground_truth["customer_id"].unique()})

    candidates = engine.generate_candidates(
        val_users, train_history, load_cached_data=False
    )

    # Validation
    expected_cols = [
        "customer_id",
        "article_id",
        "cooc_score",
        "embed_score",
        "repur_count",
        "pop_flag",
    ]
    assert all(
        col in candidates.columns for col in expected_cols
    ), f"Missing columns in candidates. Found: {candidates.columns}"
    assert len(candidates) > 0, "No candidates generated."
    print(f"Generated {len(candidates)} candidates.")

    print("\n=== Step 3: Feature Engineering ===")
    # Load auxiliary metadata
    customers_df = pd.read_csv(Config.INPUT_DIR / "customers.csv")
    articles_df = pd.read_csv(Config.INPUT_DIR / "articles.csv")

    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # Generate features
    # This merges metadata and computes dense/affinity features
    features_df = fe.generate_features(
        candidates, train_history, customers_df, articles_df, load_cached_data=False
    )

    # Validation
    assert "dense_embed_score" in features_df.columns, "Dense embedding score missing"
    assert "sales_velocity" in features_df.columns, "Trend feature missing"
    print(f"Features generated. Shape: {features_df.shape}")

    # --- Create Target Label for Ranking ---
    # We need to label the candidates as 1 (bought) or 0 (not bought) based on val_ground_truth
    print("Creating target labels...")

    # Create a set of (user, article) tuples that actually happened in validation
    actual_purchases = set(
        zip(val_ground_truth["customer_id"], val_ground_truth["article_id"])
    )

    # Apply label
    features_df["target"] = features_df.apply(
        lambda row: (
            1 if (row["customer_id"], row["article_id"]) in actual_purchases else 0
        ),
        axis=1,
    )

    print(f"Positive samples in training set: {features_df['target'].sum()}")

    print("\n=== Step 4: Ranking (LightGBM) ===")
    ranker = LGBMRanker()

    # For demonstration, we use the same dataset for train and valid to ensure it runs
    # In a real scenario, you would split `features_df` or use a separate time fold
    ranker.fit(features_df, features_df)

    # Verify model saved
    assert (Config.WORKING_DIR / "lgbm_model.txt").exists(), "LGBM model file not found"

    print("\n=== Step 5: Prediction & Submission ===")
    # Predict on the same features (acting as test set)
    submission = ranker.predict(features_df, load_model=True)

    # Validation
    assert "customer_id" in submission.columns and "prediction" in submission.columns
    assert len(submission) > 0

    # Check format of prediction string (should be space-separated IDs)
    sample_pred = submission.iloc[0]["prediction"]
    assert isinstance(sample_pred, str)
    # Should look like "012345... 012345..."
    assert len(sample_pred.split()) <= 12, "Prediction contains more than 12 items"

    print("Submission generated successfully.")
    print(submission.head())

    print("\n=== Demo Complete ===")
