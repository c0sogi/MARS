import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings
import logging

# 1. Environment Setup & Monkeypatching
# ---------------------------------------------------------
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Monkeypatch tqdm to be silent (must be done before importing modules that use it)
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import Library Modules
from library.config import (
    Paths,
    DATA_CONFIG,
    GCN_PARAMS,
    LGBM_PARAMS,
    CANDIDATE_CONFIG,
    seed_everything,
)
from library.data_loader import (
    load_raw_data,
    create_time_split,
    prepare_graph_data,
    get_recent_popular_items,
)
from library.graph_engine import train_graph_embeddings
from library.retrieval import CooccurrenceMatrix, CandidateGenerator
from library.features import FeatureEngineer
from library.ranker import Ranker


def run_demo():
    print(">>> Starting Pipeline Demo...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Configure Logger to suppress info messages for cleaner demo output
    logging.getLogger("data_loader").setLevel(logging.WARNING)
    logging.getLogger("graph_engine").setLevel(logging.WARNING)
    logging.getLogger("retrieval").setLevel(logging.WARNING)
    logging.getLogger("feature_engineering").setLevel(logging.WARNING)
    logging.getLogger("ranker").setLevel(logging.WARNING)

    # Override Hyperparameters for Speed
    print(">>> Configuring fast hyperparameters...")
    GCN_PARAMS["epochs"] = 1
    GCN_PARAMS["batch_size"] = 2048
    GCN_PARAMS["embedding_dim"] = 16  # Small dim for demo

    LGBM_PARAMS["n_estimators"] = 10
    LGBM_PARAMS["verbose"] = -1

    CANDIDATE_CONFIG["top_k_cooc"] = 6
    CANDIDATE_CONFIG["top_k_graph"] = 6
    CANDIDATE_CONFIG["top_k_repurchase"] = 6

    # 2. Data Loading & Downsampling
    # ---------------------------------------------------------
    print(">>> Loading and preprocessing data...")
    train_full, val_full, test_full = load_raw_data()

    # Downsample: Use top 500 users from the last 4 weeks of training data
    # This ensures we have a dense enough graph for the demo to work meaningfully
    last_month_start = train_full["t_dat"].max() - pd.Timedelta(days=28)
    recent_train = train_full[train_full["t_dat"] > last_month_start]

    top_users = recent_train["customer_id"].value_counts().head(500).index.tolist()

    # Filter datasets to these users
    train_subset = train_full[train_full["customer_id"].isin(top_users)].copy()
    val_subset = val_full[val_full["customer_id"].isin(top_users)].copy()
    test_subset = test_full[test_full["customer_id"].isin(top_users)].copy()

    print(
        f"    Subset Shapes -> Train: {train_subset.shape}, Val: {val_subset.shape}, Test: {test_subset.shape}"
    )

    # Create internal time split for training (Train vs Validation for Ranker)
    # We use the 'train_subset' to simulate history and 'val_subset' as the target for the ranker training
    # Note: In the library logic, 'val_subset' (from metadata) is the holdout set.

    # 3. Graph Training (LightGCN)
    # ---------------------------------------------------------
    print(">>> Training LightGCN (Graph Source)...")
    # Prepare graph data
    # Note: We disable caching to ensure we run on the subset
    edge_index, user_map, item_map = prepare_graph_data(train_subset, load_cached=False)

    num_users = len(user_map)
    num_items = len(item_map)

    # Train
    u_emb, i_emb = train_graph_embeddings(edge_index, num_users, num_items, GCN_PARAMS)

    # Validation
    assert u_emb.shape == (
        num_users,
        GCN_PARAMS["embedding_dim"],
    ), "User embedding shape mismatch"
    assert i_emb.shape == (
        num_items,
        GCN_PARAMS["embedding_dim"],
    ), "Item embedding shape mismatch"
    assert not np.isnan(u_emb).any(), "User embeddings contain NaNs"
    print("    Graph training complete.")

    # 4. Retrieval (Candidate Generation)
    # ---------------------------------------------------------
    print(">>> Generating Candidates...")

    # A. Co-occurrence Matrix
    cooc_model = CooccurrenceMatrix()
    cooc_model.fit(train_subset, load_cached=False)

    # B. Candidate Generator
    generator = CandidateGenerator(u_emb, i_emb, user_map, item_map, cooc_model)

    # Get popular items for fallback
    popular_items = get_recent_popular_items(train_subset, top_k=12)

    # Generate candidates for Validation Users
    val_customers = val_subset["customer_id"].unique().tolist()
    candidates_val = generator.generate(val_customers, train_subset, popular_items)

    # Validation
    assert "customer_id" in candidates_val.columns
    assert "article_id" in candidates_val.columns
    assert len(candidates_val) > 0, "No candidates generated"
    print(
        f"    Generated {len(candidates_val)} candidates for {len(val_customers)} validation users."
    )

    # 5. Feature Engineering
    # ---------------------------------------------------------
    print(">>> Computing Features...")

    # Load articles for metadata features
    articles_df = pd.read_csv(Paths.INPUT_DIR / "articles.csv")

    engineer = FeatureEngineer()

    # Generate features for validation candidates
    # We use train_subset as the history to compute features like velocity and graph dot product
    features_val = engineer.generate_features(
        candidates_val,
        train_subset,
        articles_df,
        u_emb,
        i_emb,
        user_map,
        item_map,
        load_cached=False,
    )

    # Validation
    expected_features = [
        "sales_velocity",
        "user_dept_ratio",
        "graph_dot_product",
        "last_item_graph_similarity",
    ]
    for f in expected_features:
        assert f in features_val.columns, f"Missing feature: {f}"
    print("    Features computed successfully.")

    # 6. Ranker Training
    # ---------------------------------------------------------
    print(">>> Training Ranker (LightGBM)...")

    # Create Labels
    # Label = 1 if the candidate article was actually bought in the validation set
    # Create a set of (user, article) tuples from ground truth
    ground_truth = set(zip(val_subset["customer_id"], val_subset["article_id"]))

    # Apply labels
    # Vectorized check
    features_val["label"] = features_val.apply(
        lambda row: 1 if (row["customer_id"], row["article_id"]) in ground_truth else 0,
        axis=1,
    )

    # Split features_val into train/val for the ranker (80/20 split of users)
    unique_val_users = features_val["customer_id"].unique()
    split_idx = int(len(unique_val_users) * 0.8)
    ranker_train_users = unique_val_users[:split_idx]

    ranker_train_df = features_val[features_val["customer_id"].isin(ranker_train_users)]
    ranker_val_df = features_val[~features_val["customer_id"].isin(ranker_train_users)]

    # Initialize Ranker
    ranker = Ranker(LGBM_PARAMS)

    # Train
    ranker.train(
        ranker_train_df,
        ranker_val_df,
        feature_cols=expected_features,
        target_col="label",
        load_cached_model=False,
    )
    print("    Ranker trained.")

    # 7. Prediction & Submission
    # ---------------------------------------------------------
    print(">>> Generating Submission for Test Subset...")

    # For the test set, we repeat the retrieval and feature engineering steps
    # 1. Generate Candidates for Test Users
    test_customers = test_subset["customer_id"].unique().tolist()

    # In a real scenario, history_df would be the full train+val.
    # Here we use train_subset + val_subset combined.
    full_history = pd.concat([train_subset, val_subset])

    candidates_test = generator.generate(test_customers, full_history, popular_items)

    # 2. Compute Features
    features_test = engineer.generate_features(
        candidates_test,
        full_history,
        articles_df,
        u_emb,
        i_emb,
        user_map,
        item_map,
        load_cached=False,
    )

    # 3. Predict Scores
    scored_test = ranker.predict(features_test, feature_cols=expected_features)

    # 4. Generate Submission File
    submission_path = Paths.WORKING_DIR / "demo_submission.csv"
    submission_df = ranker.generate_submission(scored_test, output_path=submission_path)

    # Validation
    assert os.path.exists(submission_path), "Submission file not created"
    assert len(submission_df) == len(
        test_customers
    ), "Submission incomplete (missing customers)"
    print(f"    Submission generated at {submission_path}")
    print(f"    Sample:\n{submission_df.head(2)}")

    print(">>> Demo Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_demo()
