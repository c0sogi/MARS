import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import library modules
from library import (
    config,
    data_loader,
    visual_encoder,
    graph_builder,
    retrieval,
    feature_engineering,
    ranker,
)


def main():
    print("Starting H&M Recommendation Pipeline Demo...")

    # =========================================================================
    # 1. CONFIGURATION OVERRIDES
    # =========================================================================
    print("\n[1] Configuring environment for demo...")

    # Use a specific demo directory to avoid conflicts with existing cache
    DEMO_DIR = Path("./working/demo_execution")
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch config paths
    config.WORKING_DIR = DEMO_DIR
    config.IMAGE_EMBEDDINGS_PATH = DEMO_DIR / "image_embeddings.npy"
    config.ARTICLE_ID_MAP_PATH = DEMO_DIR / "article_id_map.npy"
    config.VISUAL_GRAPH_PATH = DEMO_DIR / "visual_graph.npz"
    config.TRANSITION_MATRIX_PATH = DEMO_DIR / "transition_matrix.npz"
    config.USER_HISTORY_PATH = DEMO_DIR / "user_history.npz"
    config.RANKER_TRAIN_PATH = DEMO_DIR / "ranker_train.parquet"
    config.RANKER_VAL_PATH = DEMO_DIR / "ranker_val.parquet"
    config.OUTPUT_FILE = DEMO_DIR / "submission.csv"

    # Reduce hyperparameters for speed
    config.RECENCY_WEEKS = 4  # Reduce history window
    config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for demo
    config.LGBM_PARAMS["verbose"] = -1
    config.EARLY_STOPPING_ROUNDS = 5
    config.RETRIEVAL_TOP_K = 20  # Smaller candidate set

    # Set Seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # =========================================================================
    # 2. DATA SUBSETTING & MONKEY PATCHING
    # =========================================================================
    print("\n[2] Preparing data subset (2000 users)...")

    # Load raw metadata
    full_train = pd.read_parquet(config.TRAIN_METADATA_PATH)
    full_articles = pd.read_csv(config.ARTICLES_CSV)
    full_customers = pd.read_csv(config.CUSTOMERS_CSV)
    full_val = pd.read_parquet(config.VAL_METADATA_PATH)
    full_test = pd.read_parquet(config.TEST_METADATA_PATH)

    # Sample Users
    unique_users = full_train["customer_id"].unique()
    sampled_users = np.random.choice(unique_users, size=2000, replace=False)

    # Filter Train Transactions
    subset_train = full_train[full_train["customer_id"].isin(sampled_users)].copy()

    # Identify relevant articles (purchased by these users)
    # Add some random articles to ensure we test cases where articles aren't in history
    relevant_articles = subset_train["article_id"].unique()
    subset_articles = full_articles[
        full_articles["article_id"].isin(relevant_articles)
    ].copy()

    # Filter Customers
    subset_customers = full_customers[
        full_customers["customer_id"].isin(sampled_users)
    ].copy()

    # Filter Val Transactions
    subset_val = full_val[full_val["customer_id"].isin(sampled_users)].copy()

    # Filter Test Customers (Sample Submission)
    # We take the intersection to ensure we have history for some,
    # but we also add some "cold" users from the original test set if needed.
    # For this demo, we stick to the sampled users to verify the pipeline flow.
    subset_test = full_test[full_test["customer_id"].isin(sampled_users)].copy()

    print(
        f"Subset Stats: Users={len(sampled_users)}, Articles={len(subset_articles)}, "
        f"Train_Tx={len(subset_train)}, Val_Tx={len(subset_val)}"
    )

    # Define Wrapper Functions
    def mock_load_transactions(split="train", load_cached_data=True):
        # We ignore load_cached_data for the source, but we simulate the processing steps
        # The library function usually casts types. We do that here.
        if split == "train":
            df = subset_train.copy()
        elif split == "val":
            df = subset_val.copy()
        else:
            raise ValueError(split)

        df["t_dat"] = pd.to_datetime(df["t_dat"])
        df["article_id"] = df["article_id"].astype("int32")
        if "sales_channel_id" in df.columns:
            df["sales_channel_id"] = df["sales_channel_id"].astype("int8")
        if "price" in df.columns:
            df["price"] = df["price"].astype("float32")
        return df

    def mock_load_articles(load_cached_data=True):
        df = subset_articles.copy()
        # Add image path using the library utility
        df["image_path"] = data_loader.get_image_paths(df["article_id"])
        df["article_id"] = df["article_id"].astype("int32")
        # Categoricals
        for col in df.select_dtypes(include=["object"]).columns:
            if col != "detail_desc" and col != "image_path":
                df[col] = df[col].astype("category")
        return df

    def mock_load_customers(load_cached_data=True):
        df = subset_customers.copy()
        df["club_member_status"] = (
            df["club_member_status"].fillna("NONE").astype("category")
        )
        df["fashion_news_frequency"] = (
            df["fashion_news_frequency"].fillna("NONE").astype("category")
        )
        if "age" in df.columns:
            df["age"] = df["age"].fillna(-1).astype("int8")
        return df

    def mock_load_test_customers(load_cached_data=True):
        return subset_test.copy()

    # Apply Monkey Patches
    data_loader.load_transactions = mock_load_transactions
    data_loader.load_articles = mock_load_articles
    data_loader.load_customers = mock_load_customers
    data_loader.load_test_customers = mock_load_test_customers

    print("Monkey patching complete. Data loader now serves subsets.")

    # =========================================================================
    # 3. VISUAL ENCODER
    # =========================================================================
    print("\n[3] Running Visual Encoder...")
    # Force regeneration to use our subset
    embeddings, emb_article_ids = visual_encoder.generate_embeddings(
        load_cached_data=False, batch_size=32
    )

    # Verification
    assert len(embeddings) > 0, "Embeddings should not be empty"
    assert embeddings.shape[1] == 512, "Embedding dimension mismatch"
    assert config.IMAGE_EMBEDDINGS_PATH.exists(), "Embeddings file not saved"
    print(f"Generated embeddings for {len(embeddings)} articles.")

    # =========================================================================
    # 4. GRAPH BUILDER
    # =========================================================================
    print("\n[4] Building Graphs...")

    # Sequential Graph
    T_seq = graph_builder.build_sequential_graph(load_cached_data=False)
    assert T_seq.shape[0] == len(
        subset_articles
    ), "Adjacency matrix dimension mismatch (rows)"
    assert config.TRANSITION_MATRIX_PATH.exists(), "Transition matrix file not saved"

    # Visual Graph
    T_vis = graph_builder.build_visual_graph(load_cached_data=False)
    assert T_vis.shape == T_seq.shape, "Visual graph shape mismatch"
    assert config.VISUAL_GRAPH_PATH.exists(), "Visual graph file not saved"

    # User History
    U_hist = graph_builder.build_user_history(load_cached_data=False)
    assert U_hist.shape[0] == len(subset_customers), "User history rows mismatch"
    assert config.USER_HISTORY_PATH.exists(), "User history file not saved"

    print("Graphs built successfully.")

    # =========================================================================
    # 5. RETRIEVAL
    # =========================================================================
    print("\n[5] Testing Retrieval...")
    retriever = retrieval.SparseRetriever(load_cached_data=True)

    # Test candidate generation for a few users
    test_cids = list(subset_customers["customer_id"].values[:5])
    candidates = retriever.generate_candidates(test_cids)

    assert len(candidates) == 5, "Candidate generation count mismatch"
    for cid in test_cids:
        assert (
            len(candidates[cid]) <= config.RETRIEVAL_TOP_K
        ), "Too many candidates returned"
        assert len(candidates[cid]) > 0, "No candidates returned"

    print("Retrieval logic verified.")

    # =========================================================================
    # 6. FEATURE ENGINEERING
    # =========================================================================
    print("\n[6] Generating Ranker Dataset...")
    train_df, val_df = feature_engineering.create_ranking_dataset(
        load_cached_data=False
    )

    print(f"Ranker Train Shape: {train_df.shape}")
    print(f"Ranker Val Shape: {val_df.shape}")

    # Basic checks
    required_cols = ["label", "retrieval_score", "global_popularity"]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in training data"

    assert config.RANKER_TRAIN_PATH.exists(), "Ranker train parquet not saved"

    # =========================================================================
    # 7. RANKER TRAINING & INFERENCE
    # =========================================================================
    print("\n[7] Training Ranker and Predicting...")

    # Train
    model = ranker.train_lgbm_ranker(load_cached_model=False)
    assert model is not None, "Model training failed"

    # Predict
    ranker.predict_ranker(load_cached_data=True)

    # Verify Submission
    assert config.OUTPUT_FILE.exists(), "Submission file not created"

    sub_df = pd.read_csv(config.OUTPUT_FILE)
    assert len(sub_df) == len(subset_test), "Submission row count mismatch"
    assert (
        "customer_id" in sub_df.columns and "prediction" in sub_df.columns
    ), "Submission columns mismatch"

    # Check prediction format (space separated strings)
    sample_pred = sub_df.iloc[0]["prediction"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    items = sample_pred.split()
    assert len(items) <= 12, "More than 12 items predicted"
    # Check if items are 10-digit strings
    assert len(items[0]) == 10, "Article ID format incorrect in submission"

    print("\n[SUCCESS] Pipeline demonstration completed successfully.")
    print(f"Submission saved to: {config.OUTPUT_FILE}")


if __name__ == "__main__":
    main()
