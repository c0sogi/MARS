import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Import from the provided library
from library.config import Config
from library.data_utils import load_articles, load_customers, seed_everything
from library.visual_engine import VisualGraphBuilder
from library.graph_engine import BehavioralGraphBuilder
from library.candidate_retrieval import CandidateRetriever
from library.feature_builder import RankerDatasetGenerator
from library.ranking_model import LGBMRankerWrapper


def setup_demo_environment():
    """
    Creates a sampled subset of the data to allow the pipeline to run quickly
    for demonstration purposes. Modifies Config paths to point to this subset.
    """
    print("Setting up demo environment with sampled data...")

    # Define paths
    demo_input_dir = Path("./working/demo_input")
    demo_output_dir = Path("./working/demo_output")
    demo_sub_dir = Path("./working/demo_submission")

    if demo_input_dir.exists():
        shutil.rmtree(demo_input_dir)
    if demo_output_dir.exists():
        shutil.rmtree(demo_output_dir)
    if demo_sub_dir.exists():
        shutil.rmtree(demo_sub_dir)

    demo_input_dir.mkdir(parents=True)
    demo_output_dir.mkdir(parents=True)
    demo_sub_dir.mkdir(parents=True)

    # 1. Load original metadata to sample from
    # We use train_metadata as the source of truth for interactions
    train_df = pd.read_parquet(Config.TRAIN_METADATA)

    # Sample Top 500 Users (by activity) to ensure we have history
    top_users = train_df["customer_id"].value_counts().head(500).index.tolist()
    sampled_train = train_df[train_df["customer_id"].isin(top_users)].copy()

    # Identify relevant articles from these transactions
    relevant_articles = sampled_train["article_id"].unique().tolist()

    # Also include some random articles to test retrieval of non-interacted items
    # (Total approx 1000-2000 articles)

    # 2. Filter Articles CSV
    articles_df = pd.read_csv(Config.ARTICLES_CSV)
    sampled_articles_df = articles_df[
        articles_df["article_id"].isin(relevant_articles)
    ].copy()

    # 3. Filter Customers CSV
    customers_df = pd.read_csv(Config.CUSTOMERS_CSV)
    sampled_customers_df = customers_df[
        customers_df["customer_id"].isin(top_users)
    ].copy()

    # 4. Create Sampled Metadata Files
    # Split sampled_train into train/val for the demo
    # Sort by date
    sampled_train = sampled_train.sort_values("t_dat")

    # Simple split: Last 7 days of the sampled data as 'val', rest as 'train'
    # Note: Real pipeline uses pre-defined split, here we simulate it for the demo subset
    max_date = pd.to_datetime(sampled_train["t_dat"]).max()
    split_date = max_date - pd.Timedelta(days=7)

    sampled_train["t_dat"] = pd.to_datetime(sampled_train["t_dat"])

    demo_train = sampled_train[sampled_train["t_dat"] < split_date].copy()
    demo_val = sampled_train[sampled_train["t_dat"] >= split_date].copy()

    # Create Test Set (Sample Submission)
    # Just the list of users we sampled
    demo_test = pd.DataFrame({"customer_id": top_users, "prediction": ""})

    # 5. Save Files
    p_articles = demo_input_dir / "articles.csv"
    p_customers = demo_input_dir / "customers.csv"
    p_train = demo_input_dir / "train_metadata.parquet"
    p_val = demo_input_dir / "val_metadata.parquet"
    p_test = demo_input_dir / "test_metadata.parquet"
    p_sample_sub = demo_input_dir / "sample_submission.csv"

    sampled_articles_df.to_csv(p_articles, index=False)
    sampled_customers_df.to_csv(p_customers, index=False)
    demo_train.to_parquet(p_train, index=False)
    demo_val.to_parquet(p_val, index=False)
    demo_test.to_parquet(p_test, index=False)
    demo_test.to_csv(
        p_sample_sub, index=False
    )  # Needs CSV format for Config.SAMPLE_SUBMISSION_CSV

    print(
        f"Sampled Data Stats: Users={len(sampled_customers_df)}, Articles={len(sampled_articles_df)}"
    )

    # 6. Monkey-Patch Config
    print("Updating Config paths...")
    Config.WORKING_DIR = demo_output_dir
    Config.SUBMISSION_DIR = demo_sub_dir

    Config.ARTICLES_CSV = p_articles
    Config.CUSTOMERS_CSV = p_customers
    Config.TRAIN_METADATA = p_train
    Config.VAL_METADATA = p_val
    Config.TEST_METADATA = p_test
    Config.SAMPLE_SUBMISSION_CSV = p_sample_sub

    # Update Artifact Paths to be inside the new working dir
    Config.ARTICLE_ID_MAP_PATH = demo_output_dir / "article_id_map.npy"
    Config.CUSTOMER_ID_MAP_PATH = demo_output_dir / "customer_id_map.npy"
    Config.ARTICLE_EMBEDDINGS_PATH = demo_output_dir / "article_embeddings.npy"
    Config.VISUAL_KNN_GRAPH_PATH = demo_output_dir / "visual_knn_graph.npz"
    Config.TRANSITION_MATRIX_PATH = demo_output_dir / "transition_matrix.npz"
    Config.GLOBAL_POPULARITY_PATH = demo_output_dir / "global_popularity.parquet"
    Config.RANKER_TRAIN_SET = demo_output_dir / "ranker_train_set.parquet"
    Config.RANKER_VAL_SET = demo_output_dir / "ranker_val_set.parquet"
    Config.RANKER_MODEL_PATH = demo_output_dir / "lgbm_ranker.txt"
    Config.SUBMISSION_PATH = demo_sub_dir / "submission.csv"

    # Optimize Hyperparameters for Speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_round"] = 5
    Config.SLIDING_WINDOW_WEEKS = 2  # Only process last 2 weeks
    Config.RETRIEVAL_TOP_K = 20  # Smaller candidate set
    Config.RANKING_TOP_K = 12
    Config.BATCH_SIZE = 32  # Smaller batch for visual engine


def run_pipeline():
    seed_everything(Config.SEED)

    # --- Step 1: Data Loading & Preprocessing ---
    print("\n[Step 1] Loading and Preprocessing Data...")
    # This generates the dense index maps and saves them to the working dir
    articles_df, art_map = load_articles(load_cached_data=False)
    customers_df, cust_map = load_customers(load_cached_data=False)

    assert len(articles_df) > 0, "Articles dataframe is empty"
    assert len(customers_df) > 0, "Customers dataframe is empty"
    assert Config.ARTICLE_ID_MAP_PATH.exists(), "Article map not saved"
    print("Data loaded successfully.")

    # --- Step 2: Visual Engine (Embeddings & KNN) ---
    print("\n[Step 2] Running Visual Engine...")
    vis_builder = VisualGraphBuilder()

    # Extract embeddings (will use the sampled articles)
    embeddings = vis_builder.extract_embeddings(load_cached_data=False)
    assert embeddings.shape[0] == len(articles_df), "Embedding count mismatch"
    assert embeddings.shape[1] == Config.EMBEDDING_DIM, "Embedding dimension mismatch"

    # Build KNN Graph
    knn_graph = vis_builder.build_knn_graph(load_cached_data=False)
    assert knn_graph.shape == (
        len(articles_df),
        len(articles_df),
    ), "KNN Graph shape mismatch"
    print("Visual components built successfully.")

    # --- Step 3: Graph Engine (Behavioral Matrices) ---
    print("\n[Step 3] Running Behavioral Graph Engine...")
    beh_builder = BehavioralGraphBuilder()

    # Transition Matrix
    trans_matrix = beh_builder.build_transition_matrix(load_cached_data=False)
    assert trans_matrix.shape == (
        len(articles_df),
        len(articles_df),
    ), "Transition Matrix shape mismatch"

    # Global Popularity
    pop_df = beh_builder.build_global_popularity(load_cached_data=False)
    assert not pop_df.empty, "Global popularity is empty"
    print("Behavioral components built successfully.")

    # --- Step 4: Feature Generation (Ranker Data) ---
    print("\n[Step 4] Generating Ranker Datasets...")
    dataset_gen = RankerDatasetGenerator()

    # This generates train and val parquet files in the working dir
    dataset_gen.generate_sliding_window_data(load_cached_data=False)

    assert Config.RANKER_TRAIN_SET.exists(), "Ranker Train Set not created"
    assert Config.RANKER_VAL_SET.exists(), "Ranker Val Set not created"

    # Verify content
    train_check = pd.read_parquet(Config.RANKER_TRAIN_SET)
    print(f"Generated {len(train_check)} training samples.")
    print("Feature generation complete.")

    # --- Step 5: Ranking Model (Training) ---
    print("\n[Step 5] Training Ranker...")
    ranker = LGBMRankerWrapper()

    ranker.train(load_cached_data=False)

    assert Config.RANKER_MODEL_PATH.exists(), "Model file not saved"
    print("Model trained successfully.")

    # --- Step 6: Submission Generation ---
    print("\n[Step 6] Generating Submission...")
    ranker.generate_submission(load_cached_data=False)

    assert Config.SUBMISSION_PATH.exists(), "Submission file not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "customer_id" in sub_df.columns
    assert "prediction" in sub_df.columns
    assert len(sub_df) > 0

    # Check if predictions look like article IDs (10 digits)
    sample_pred = sub_df.iloc[0]["prediction"]
    if pd.notna(sample_pred) and len(sample_pred) > 0:
        first_item = sample_pred.split()[0]
        assert len(first_item) == 10, f"Invalid article ID format: {first_item}"

    print(f"Submission generated with {len(sub_df)} rows.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure we don't accidentally overwrite real work if run locally,
    # though the prompt environment is isolated.
    try:
        setup_demo_environment()
        run_pipeline()
    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
