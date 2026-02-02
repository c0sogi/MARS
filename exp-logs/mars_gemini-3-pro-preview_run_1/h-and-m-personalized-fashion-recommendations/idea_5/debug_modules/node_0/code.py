import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
import shutil
from datetime import datetime, timedelta

# Import the provided library modules
from library import config
from library import data_processor
from library import graph_engine
from library import inference_model
from library import evaluation


# =============================================================================
# 1. SETUP DEMO ENVIRONMENT (Data Subsetting)
# =============================================================================
def setup_demo_data(base_dir):
    """
    Creates a consistent small subset of data in base_dir for demonstration.
    """
    print(f"Setting up demo data in {base_dir}...")
    os.makedirs(base_dir, exist_ok=True)

    # Define paths for demo files
    demo_train_path = os.path.join(base_dir, "train.csv")
    demo_val_path = os.path.join(base_dir, "val.csv")
    demo_test_path = os.path.join(base_dir, "test.csv")
    demo_articles_path = os.path.join(base_dir, "articles.csv")

    # 1. Load a small sample of Articles
    # We need enough columns for the variant matrix (product_code)
    print("Sampling articles...")
    full_articles = pd.read_csv(config.ARTICLES_PATH)
    # Take top 500 articles
    demo_articles = full_articles.head(500).copy()
    demo_articles.to_csv(demo_articles_path, index=False)
    valid_article_ids = set(demo_articles["article_id"])

    # 2. Load a sample of Train Transactions
    print("Sampling training transactions...")
    # Load first 10k rows, filter for valid articles
    full_train = pd.read_csv(config.TRAIN_DATA_PATH, nrows=20000)
    demo_train = full_train[full_train["article_id"].isin(valid_article_ids)].copy()

    # Ensure we have some data
    if len(demo_train) == 0:
        # Fallback: Create synthetic data if intersection is empty
        print("Warning: No intersection found. Creating synthetic transactions.")
        demo_train = full_train.head(100).copy()
        demo_train["article_id"] = list(valid_article_ids)[:100]

    demo_train.to_csv(demo_train_path, index=False)
    valid_customers = set(demo_train["customer_id"])

    # 3. Create Validation set (using same customers to ensure history exists)
    print("Creating validation set...")
    full_val = pd.read_csv(config.VAL_DATA_PATH, nrows=5000)
    # Filter or force overlap
    demo_val = full_val[full_val["customer_id"].isin(valid_customers)].copy()
    if len(demo_val) < 10:
        # Create synthetic val data from train users
        demo_val = demo_train.head(20).copy()
        # Shift dates for validation
        demo_val["t_dat"] = "2020-09-23"

    # Ensure article IDs are valid in val
    demo_val = demo_val[demo_val["article_id"].isin(valid_article_ids)]
    demo_val.to_csv(demo_val_path, index=False)

    # 4. Create Test set
    print("Creating test set...")
    # Just use the unique customers from our demo train/val
    all_demo_custs = list(valid_customers)
    demo_test = pd.DataFrame(
        {"customer_id": all_demo_custs[:50]}
    )  # Predict for 50 users
    demo_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path, demo_articles_path


# =============================================================================
# 2. MONKEY PATCH CONFIGURATION
# =============================================================================
def patch_config(demo_paths, working_dir):
    """
    Overrides library.config paths to point to the demo data.
    """
    train_p, val_p, test_p, art_p = demo_paths

    # Override input paths
    config.TRAIN_DATA_PATH = train_p
    config.VAL_DATA_PATH = val_p
    config.TEST_DATA_PATH = test_p
    config.ARTICLES_PATH = art_p

    # Override working directory to avoid messing with real cache
    config.WORKING_DIR = working_dir
    os.makedirs(working_dir, exist_ok=True)

    # Update cache paths based on new working dir
    config.CACHE_TRANSACTIONS_PROCESSED = os.path.join(
        working_dir, "transactions_processed.parquet"
    )
    config.CACHE_MATRICES_HYBRID = os.path.join(
        working_dir, "hybrid_similarity_matrix.npz"
    )
    config.CACHE_USER_HISTORY = os.path.join(working_dir, "user_history_vectors.npz")
    config.CACHE_GLOBAL_TRENDS = os.path.join(working_dir, "global_trends.parquet")
    config.CACHE_ITEM_MAP = os.path.join(working_dir, "item_id_map.parquet")
    config.CACHE_USER_MAP = os.path.join(working_dir, "user_id_map.parquet")
    config.SUBMISSION_PATH = os.path.join(working_dir, "submission_demo.csv")

    # Reduce batch size for demo
    config.BATCH_SIZE = 10


# =============================================================================
# 3. DEMONSTRATION MODULES
# =============================================================================


def demo_data_loader():
    print("\n=== DEMO: Data Processor ===")
    loader = data_processor.DataLoader()

    # Force recompute to test logic (load_cached_data=False)
    train_df, val_df, test_df, articles_df, user_map, item_map = loader.load_data(
        load_cached_data=False
    )

    print(f"Loaded Train Shape: {train_df.shape}")
    print(f"User Map Size: {len(user_map)}")
    print(f"Item Map Size: {len(item_map)}")

    # Validation
    assert "user_idx" in train_df.columns, "user_idx missing from processed train"
    assert "item_idx" in train_df.columns, "item_idx missing from processed train"
    assert len(item_map) > 0, "Item map is empty"

    return train_df, articles_df, user_map, item_map


def demo_graph_engine(train_df, articles_df, user_map, item_map):
    print("\n=== DEMO: Graph Engine ===")
    optimizer = graph_engine.SimilarityOptimizer()

    # Run the graph construction pipeline
    S_hybrid = optimizer.run(
        train_df, articles_df, user_map, item_map, load_cached_data=False
    )

    print(f"Hybrid Matrix Shape: {S_hybrid.shape}")
    print(f"Hybrid Matrix Non-zeros: {S_hybrid.nnz}")

    # Validation
    assert S_hybrid.shape == (
        len(item_map),
        len(item_map),
    ), "Similarity matrix shape mismatch"
    assert S_hybrid.nnz > 0, "Similarity matrix is empty"

    return S_hybrid


def demo_inference_and_evaluation():
    print("\n=== DEMO: Inference & Evaluation ===")

    # Instantiate the Recommender
    recommender = inference_model.StratifiedRecommender()

    # Load resources (this will use the cache generated in previous steps if available,
    # or recompute. We set load_cached_data=True to use what we just built)
    _, val_df, test_df = recommender.load_resources(load_cached_data=True)

    # Generate predictions for the test set
    print(f"Predicting for {len(test_df)} test users...")
    submission_df = recommender.predict(test_df, k=12)

    print("Sample Predictions:")
    print(submission_df.head(3))

    # Validation of Output Format
    assert "customer_id" in submission_df.columns
    assert "prediction" in submission_df.columns
    # Check prediction string format
    first_pred = submission_df.iloc[0]["prediction"]
    assert isinstance(first_pred, str)
    pred_items = first_pred.split()
    assert len(pred_items) <= 12, "Predicted more than 12 items"

    # Evaluate using the provided metric
    # We use the validation set (which we ensured has history) to calculate MAP@12
    # We need to predict for validation users first
    print("Predicting for validation users to compute MAP@12...")
    val_preds_df = recommender.predict(val_df[["customer_id"]].drop_duplicates(), k=12)

    map_score = evaluation.calculate_map12(val_df, val_preds_df, k=12)
    print(f"Computed MAP@12 on Demo Validation Set: {map_score:.6f}")

    assert 0.0 <= map_score <= 1.0, "MAP score out of range"


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Fix seeds
    np.random.seed(42)

    # Define paths
    demo_dir = "./working/demo_run"

    try:
        # 1. Setup
        paths = setup_demo_data(demo_dir)
        patch_config(paths, demo_dir)

        # 2. Data Processing
        train_df, articles_df, user_map, item_map = demo_data_loader()

        # 3. Graph Construction
        S_hybrid = demo_graph_engine(train_df, articles_df, user_map, item_map)

        # 4. Inference & Eval
        demo_inference_and_evaluation()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise e
