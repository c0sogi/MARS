import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_map12
from library.data_loader import DataLoader
from library.visual_encoder import VisualEncoder
from library.graph_engine import GraphEngine
from library.retrieval_system import SparseRetriever
from library.feature_engineering import FeatureEngine
from library.ranker import LGBMRanker


def run_demo():
    print("Initializing H&M Recommender System Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset for speed
    Config.LGBM_PARAMS["n_estimators"] = 10  # Minimal trees for training
    Config.LGBM_PARAMS["verbose"] = -1
    Config.NUM_WORKERS = 2  # Reduce workers for demo environment

    # Ensure working directory is clean-ish or just overwrite
    # We force load_cached_data=False to demonstrate logic execution

    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and Preprocessing Data...")
    loader = DataLoader()

    # Force processing from raw files
    train_df, val_df, test_df, articles_df, customers_df = loader.load_data(
        load_cached_data=False
    )

    # Validation
    assert not train_df.empty, "Train DataFrame is empty"
    assert not val_df.empty, "Val DataFrame is empty"
    assert "article_id" in train_df.columns, "article_id column missing in train"
    assert train_df["customer_id"].dtype == "int32", "Customer IDs not mapped to int32"
    print(f"Data Loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Visual Encoding
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating Visual Embeddings...")
    encoder = VisualEncoder()

    # Force generation
    embeddings = encoder.generate_embeddings(load_cached_data=False)

    # Validation
    expected_articles = len(np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True))
    assert (
        embeddings.shape[0] == expected_articles
    ), f"Embedding count mismatch. Expected {expected_articles}, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 512, "Embedding dimension mismatch. Expected 512."
    print(f"Embeddings Generated. Shape: {embeddings.shape}")

    # -------------------------------------------------------------------------
    # 4. Graph Construction
    # -------------------------------------------------------------------------
    print("\n[Step 3] Building Sparse Graphs...")
    ge = GraphEngine()
    ge.build_graphs(train_df, embeddings, load_cached_data=False)

    # Validation
    assert Config.CACHE_GRAPH_SHORT.exists(), "Short-term graph file missing"
    assert Config.CACHE_GRAPH_LONG.exists(), "Long-term graph file missing"
    assert Config.CACHE_GRAPH_VISUAL.exists(), "Visual graph file missing"
    assert Config.CACHE_USER_HISTORY.exists(), "User history graph file missing"
    print("Graph construction complete.")

    # -------------------------------------------------------------------------
    # 5. Retrieval & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 4] Retrieval and Feature Engineering...")
    retriever = SparseRetriever()
    fe = FeatureEngine()

    # A. Generate Training Data (Sliding Windows)
    print("  - Generating Training Data...")
    ranker_train = fe.generate_train_data(
        retriever, loader, train_df, articles_df, customers_df, load_cached_data=False
    )
    assert not ranker_train.empty, "Ranker training data is empty"
    assert "label" in ranker_train.columns, "Label column missing in training data"

    # B. Generate Validation Data
    print("  - Generating Validation Data...")
    ranker_val = fe.generate_val_data(
        retriever, val_df, articles_df, customers_df, load_cached_data=False
    )
    assert not ranker_val.empty, "Ranker validation data is empty"

    # -------------------------------------------------------------------------
    # 6. Model Training
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training LightGBM Ranker...")
    ranker = LGBMRanker()
    ranker.train(ranker_train, ranker_val)

    assert ranker.model is not None, "Model failed to train"
    print("Model trained successfully.")

    # -------------------------------------------------------------------------
    # 7. Evaluation (MAP@12 on Validation Set)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Evaluating on Validation Set...")
    # Predict scores
    val_scored = ranker.predict(ranker_val)

    # To use the calculate_map12 utility, we need to format predictions into strings
    # We reuse the logic from generate_submission but keep it in memory

    # 1. Get Top 12 per customer
    top_k_val = val_scored.sort_values(
        ["customer_id", "prediction_score"], ascending=[True, False]
    )
    top_k_val = top_k_val.groupby("customer_id").head(12)

    # 2. Map IDs back to strings
    article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
    customer_map = np.load(Config.CACHE_CUSTOMER_MAP, allow_pickle=True)

    # Helper to convert list of ints to space-separated string
    def ids_to_string(ids):
        return " ".join([f"{article_map[i]:010d}" for i in ids])

    preds_series = top_k_val.groupby("customer_id")["article_id"].apply(ids_to_string)

    # 3. Create DataFrame expected by calculate_map12
    # Map customer index back to original ID for the metric function if needed,
    # but calculate_map12 works with whatever ID is in the column as long as it matches ground truth.
    # Since val_df uses mapped integers, we can stick to integers for the join,
    # BUT calculate_map12 expects article_ids in ground truth to be convertible to string.

    # Let's construct the prediction DF with mapped integer customer_ids (as index/col)
    preds_df = preds_series.reset_index()
    preds_df.columns = ["customer_id", "prediction"]

    # Ground Truth: val_df has ['customer_id', 'article_id'] (integers)
    # We need to map article_ids in val_df to strings to match the prediction format
    val_ground_truth = val_df.copy()
    val_ground_truth["article_id"] = val_ground_truth["article_id"].apply(
        lambda x: f"{article_map[x]:010d}"
    )

    # Calculate MAP
    map_score = calculate_map12(preds_df, val_ground_truth)
    print(f"Validation MAP@12: {map_score:.6f}")

    # -------------------------------------------------------------------------
    # 8. Final Submission Generation (Test Set)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Test Submission...")

    # Generate Test Features
    # Note: test_df contains the target customers for submission
    ranker_test = fe.generate_test_data(
        retriever,
        train_df,
        val_df,
        test_df,
        articles_df,
        customers_df,
        load_cached_data=False,
    )

    if ranker_test.empty:
        print(
            "Warning: No candidates retrieved for test set (expected in small debug sample)."
        )
        # Create dummy scored df to test submission logic
        ranker_test = pd.DataFrame(
            {
                "customer_id": test_df["customer_id"].unique(),
                "article_id": [0] * test_df["customer_id"].nunique(),  # Dummy article 0
                "prediction_score": [0.0] * test_df["customer_id"].nunique(),
                **{
                    c: [0] * test_df["customer_id"].nunique()
                    for c in ranker.feature_cols
                },
            }
        )
        # Need to ensure columns match for predict
        test_scored = ranker_test  # Skip predict if dummy
        test_scored["prediction_score"] = 0.5
    else:
        test_scored = ranker.predict(ranker_test)

    # Generate File
    ranker.generate_submission(test_scored, test_df)

    assert Config.SUBMISSION_PATH.exists(), "Submission file was not created"

    # Verify file content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "customer_id" in sub_df.columns
    assert "prediction" in sub_df.columns
    assert len(sub_df) > 0

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("Demo Execution Complete.")


if __name__ == "__main__":
    run_demo()
