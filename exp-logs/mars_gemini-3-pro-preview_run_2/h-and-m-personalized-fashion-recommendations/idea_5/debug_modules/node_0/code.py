import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import sequential_encoder
from library import heuristics
from library import candidate_generation
from library import feature_engineering
from library import ranking

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting H&M Recommendation Pipeline Demo ===")

    # 1. Setup and Reproducibility
    utils.seed_everything(config.RANDOM_STATE)

    # --- Override Config for Speed ---
    print("Overriding configuration for fast demonstration...")
    config.SEQ_CONFIG["epochs"] = 1
    config.SEQ_CONFIG["batch_size"] = 256
    config.SEQ_CONFIG["embedding_dim"] = 64  # Smaller dim
    config.SEQ_CONFIG["n_heads"] = 2
    config.SEQ_CONFIG["n_layers"] = 1

    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 16
    config.LGBM_PARAMS["verbose"] = -1

    # Reduce retrieval limits
    config.COOC_TOP_K = 12
    config.REPURCHASE_LIMIT = 12

    # 2. Data Loading & Subsetting
    print("\n[Step 1] Loading and Subsetting Data...")
    df_history, df_articles, df_customers = data_loader.load_raw_data()

    # Filter to top 500 active customers to ensure density and speed
    top_customers = df_history["customer_id"].value_counts().head(500).index.tolist()
    df_subset = df_history[df_history["customer_id"].isin(top_customers)].copy()

    # Also filter customers metadata
    df_customers_subset = df_customers[
        df_customers["customer_id"].isin(top_customers)
    ].copy()

    print(f"Subset history shape: {df_subset.shape}")
    print(f"Subset customers: {df_customers_subset.shape[0]}")

    # 3. Time Split
    print("\n[Step 2] Splitting Data (Train vs Validation)...")
    # Using default VAL_DAYS from config (7 days)
    train_df, val_df = data_loader.get_time_split(df_subset)

    # We will generate candidates for users present in the validation set
    target_val_users = val_df["customer_id"].unique()
    print(f"Target validation users: {len(target_val_users)}")

    # 4. Sequential Model
    print("\n[Step 3] Training Sequential Model...")
    # Preprocess
    seq_data = data_loader.preprocess_sequences(
        train_df,
        min_history=2,  # Lower min_history for demo
        max_seq_len=config.SEQ_CONFIG["max_seq_len"],
    )

    # Train
    seq_model = sequential_encoder.train_sequential_model(
        seq_data, params=config.SEQ_CONFIG, load_cached_data=False  # Force retrain
    )

    # Extract Embeddings
    user_embeddings, item_embeddings = sequential_encoder.extract_embeddings(
        seq_model, seq_data
    )

    # Validation
    assert (
        user_embeddings.shape[1] == config.SEQ_CONFIG["embedding_dim"]
    ), "User embedding dim mismatch"
    assert (
        item_embeddings.shape[1] == config.SEQ_CONFIG["embedding_dim"]
    ), "Item embedding dim mismatch"
    print("Sequential model trained and embeddings extracted successfully.")

    # 5. Co-occurrence Matrix (Explicit Fit for Feature Engineering)
    print("\n[Step 4] Fitting Co-occurrence Matrix...")
    cooc_matrix = heuristics.CooccurrenceMatrix()
    cooc_matrix.fit(train_df, load_cached_data=False)

    assert cooc_matrix.is_fitted, "Co-occurrence matrix failed to fit"
    print("Co-occurrence matrix fitted.")

    # 6. Candidate Generation
    print("\n[Step 5] Generating Candidates...")
    orchestrator = candidate_generation.CandidateOrchestrator()

    # Generate candidates for validation users using training history
    # We set load_cached_data=False to ensure logic runs
    candidates_df = orchestrator.generate_candidates(
        history_df=train_df,
        target_customer_ids=target_val_users,
        cache_path=config.WORKING_DIR / "demo_candidates.parquet",
        load_cached_data=False,
    )

    assert not candidates_df.empty, "Candidate generation returned empty DataFrame"
    assert (
        "customer_id" in candidates_df.columns and "article_id" in candidates_df.columns
    )
    print(f"Generated {len(candidates_df)} unique candidates.")

    # 7. Feature Engineering
    print("\n[Step 6] Computing Features...")
    feature_builder = feature_engineering.FeatureBuilder()

    features_df = feature_builder.compute_features(
        candidates_df=candidates_df,
        history_df=train_df,
        articles_df=df_articles,
        customers_df=df_customers_subset,
        seq_data=seq_data,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        cooc_matrix=cooc_matrix,
        load_cached_data=False,
        cache_name="demo_features.parquet",
    )

    # Validate Features
    expected_cols = ["sales_trend", "seq_similarity", "cooc_score", "purchase_count"]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature column: {col}"

    # Check for NaNs in critical numerical features
    assert not features_df["cooc_score"].isnull().any(), "NaNs found in cooc_score"
    print(f"Feature matrix shape: {features_df.shape}")

    # 8. Ranking
    print("\n[Step 7] Training Ranker...")
    ranker = ranking.LGBMRankerWrapper()

    # Split candidates into Train/Val for the ranker
    # For this demo, we'll split the users in the validation set into two halves
    unique_users = features_df["customer_id"].unique()
    np.random.shuffle(unique_users)
    split_idx = int(len(unique_users) * 0.8)

    ranker_train_users = unique_users[:split_idx]
    ranker_val_users = unique_users[split_idx:]

    train_cands = features_df[
        features_df["customer_id"].isin(ranker_train_users)
    ].copy()
    val_cands = features_df[features_df["customer_id"].isin(ranker_val_users)].copy()

    # Ground Truth is the actual purchases in val_df
    train_gt = val_df[val_df["customer_id"].isin(ranker_train_users)].copy()
    val_gt = val_df[val_df["customer_id"].isin(ranker_val_users)].copy()

    # Define feature columns (exclude IDs and non-numeric metadata if needed,
    # but LGBM handles categories if encoded. The pipeline encoded them.)
    # We exclude object types just to be safe for this generic demo,
    # though FeatureBuilder encodes most.
    feature_cols = [
        c
        for c in features_df.columns
        if c not in ["customer_id", "article_id", "t_dat"]
    ]

    # Train
    ranker.train(
        train_candidates=train_cands,
        val_candidates=val_cands,
        train_ground_truth=train_gt,
        val_ground_truth=val_gt,
        feature_cols=feature_cols,
        params=config.LGBM_PARAMS,
        load_cached_model=False,
    )

    # 9. Prediction & Evaluation
    print("\n[Step 8] Predicting and Evaluating...")
    # Predict on the validation set (simulating test)
    submission_df = ranker.predict(val_cands, feature_cols)

    assert "customer_id" in submission_df.columns
    assert "prediction" in submission_df.columns

    # Calculate Score
    score = utils.calculate_map12(val_gt, submission_df)
    print(f"\nFinal Demo Score (MAP@12) on Validation Subset: {score:.6f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
