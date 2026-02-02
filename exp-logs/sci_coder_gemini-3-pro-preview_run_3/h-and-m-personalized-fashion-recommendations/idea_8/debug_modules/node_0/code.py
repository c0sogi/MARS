import os
import shutil
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from pathlib import Path

# Import library modules
import library.config as config
import library.data_manager as data_manager
import library.visual_module as visual_module
import library.retrieval_engine as retrieval_engine
import library.feature_engine as feature_engine
import library.ranking_model as ranking_model


# =============================================================================
# SETUP & CONFIGURATION OVERRIDES
# =============================================================================
def setup_reproducibility():
    """Sets random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def override_config_for_demo():
    """
    Dynamically overrides config parameters to run a fast demo
    on a subset of data without modifying the library files.
    """
    # Define demo directories
    demo_base = Path("./working/demo_execution")
    demo_input = demo_base / "inputs"
    demo_artifacts = demo_base / "artifacts"

    # Clean up previous run
    if demo_base.exists():
        shutil.rmtree(demo_base)
    demo_input.mkdir(parents=True, exist_ok=True)
    demo_artifacts.mkdir(parents=True, exist_ok=True)

    print(f"Configuring demo environment at {demo_base}...")

    # Override File Paths in config
    config.ARTICLES_CSV = demo_input / "articles.csv"
    config.CUSTOMERS_CSV = demo_input / "customers.csv"
    config.TRAIN_METADATA = demo_input / "train_metadata.parquet"
    config.VAL_METADATA = demo_input / "val_metadata.parquet"
    config.TEST_METADATA = demo_input / "test_metadata.parquet"

    config.WORKING_DIR = demo_artifacts
    config.ARTICLE_ID_MAP_PATH = demo_artifacts / "article_id_map.npy"
    config.CUSTOMER_ID_MAP_PATH = demo_artifacts / "customer_id_map.npy"
    config.IMAGE_EMBEDDINGS_PATH = demo_artifacts / "image_embeddings.npy"
    config.VISUAL_GRAPH_PATH = demo_artifacts / "visual_graph.npz"
    config.SEQUENTIAL_GRAPH_PATH = demo_artifacts / "sequential_graph.npz"
    config.LGBM_MODEL_PATH = demo_artifacts / "lgbm_model.txt"
    config.SUBMISSION_CSV = demo_artifacts / "submission.csv"

    # Override Hyperparameters for Speed
    config.IMAGE_BATCH_SIZE = 32
    config.VISUAL_KNN_K = 5
    config.TOP_K_CANDIDATES = 20
    config.RANKER_WINDOW_COUNT = 1  # Only 1 fold
    config.HISTORY_WINDOW_SIZE = 4  # Look back 4 weeks

    # LightGBM Speedups
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8
    config.LGBM_PARAMS["min_data_in_leaf"] = 5
    config.LGBM_PARAMS["verbose"] = -1

    return demo_input


def create_demo_data(demo_input_dir):
    """
    Creates a consistent subset of data (Mini-Dataset) for the demo.
    """
    print("Creating subsampled dataset for demo...")

    # 1. Load original metadata
    # We use the provided metadata files as the source
    orig_train = pd.read_parquet("./metadata/train_metadata.parquet")

    # 2. Sample Top N Customers and Articles to ensure density
    # Take top 200 customers by transaction count
    top_customers = orig_train["customer_id"].value_counts().head(200).index.tolist()

    # Filter transactions for these customers
    subset_df = orig_train[orig_train["customer_id"].isin(top_customers)].copy()

    # Take top 500 articles from these transactions
    top_articles = subset_df["article_id"].value_counts().head(500).index.tolist()

    # Final Filter
    subset_df = subset_df[subset_df["article_id"].isin(top_articles)].copy()

    print(
        f"Demo Dataset: {len(subset_df)} transactions, {len(top_customers)} users, {len(top_articles)} items."
    )

    # 3. Create Metadata Parquets
    # Split subset into train/val/test for the demo flow
    # Sort by date
    subset_df = subset_df.sort_values("t_dat")

    # Simple time split: Last 7 days as Val, rest as Train
    max_date = subset_df["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=7)

    train_split = subset_df[subset_df["t_dat"] < split_date].copy()
    val_split = subset_df[subset_df["t_dat"] >= split_date].copy()

    # Test set is just a list of customers to predict for
    test_split = pd.DataFrame({"customer_id": top_customers})
    test_split["prediction"] = ""  # Placeholder

    # Save Parquets
    train_split.to_parquet(config.TRAIN_METADATA, index=False)
    val_split.to_parquet(config.VAL_METADATA, index=False)
    test_split.to_parquet(config.TEST_METADATA, index=False)

    # 4. Create Raw CSVs (articles.csv, customers.csv)
    # These are needed because data_manager.get_id_mappings reads raw CSVs

    # Load original raw files to get metadata columns
    orig_articles = pd.read_csv("./input/articles.csv")
    orig_customers = pd.read_csv("./input/customers.csv")

    # Filter
    demo_articles = orig_articles[orig_articles["article_id"].isin(top_articles)]
    demo_customers = orig_customers[orig_customers["customer_id"].isin(top_customers)]

    # Save CSVs
    demo_articles.to_csv(config.ARTICLES_CSV, index=False)
    demo_customers.to_csv(config.CUSTOMERS_CSV, index=False)

    print("Demo data files created successfully.")


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
if __name__ == "__main__":
    setup_reproducibility()

    # 1. Prepare Environment
    demo_input_dir = override_config_for_demo()
    create_demo_data(demo_input_dir)

    # 2. Data Loading & Mapping
    print("\n--- Step 2: ID Mapping ---")
    # Force re-creation of mappings from our new mini CSVs
    cust_to_idx, idx_to_cust, art_to_idx, idx_to_art = data_manager.get_id_mappings(
        load_cached_data=False
    )

    assert len(idx_to_cust) <= 200, "Customer mapping should reflect subsample"
    assert len(idx_to_art) <= 500, "Article mapping should reflect subsample"

    # Load the parquet metadata
    train_df, val_df, test_df = data_manager.load_metadata()

    # 3. Visual Module
    print("\n--- Step 3: Visual Module (Embeddings & Graph) ---")
    # This will run ResNet on the ~500 images (or zeros if missing)
    # and build a KNN graph.
    visual_graph = visual_module.build_visual_graph(load_cached_data=False)

    assert visual_graph.shape == (
        len(idx_to_art),
        len(idx_to_art),
    ), "Visual graph shape mismatch"

    # 4. Retrieval Engine
    print("\n--- Step 4: Retrieval ---")
    retriever = retrieval_engine.DualViewRetriever()

    # Build Sequential Graph from Training Data
    seq_graph = retriever.build_sequential_graph(
        train_df, cache_key="demo", load_cached_data=False
    )

    # Build User History Vectors for Test Customers
    # In this demo, we predict for the customers in our test_split
    target_customers = test_df["customer_id"].unique()
    user_vectors = retriever.build_user_vectors(train_df, target_customers)

    # Retrieve Candidates
    candidates_df = retriever.retrieve(
        user_vectors=user_vectors,
        sequential_graph=seq_graph,
        visual_graph=visual_graph,
        customer_ids=target_customers,
    )

    print(
        f"Retrieved {len(candidates_df)} candidates for {len(target_customers)} users."
    )
    assert not candidates_df.empty, "Retrieval returned no candidates"
    assert "score_vis" in candidates_df.columns, "Missing visual score"

    # 5. Feature Engineering
    print("\n--- Step 5: Feature Engineering ---")
    feature_gen = feature_engine.FeatureGenerator()

    # Generate features for the candidates
    # We treat 'train_df' as history.
    # For this demo, we won't generate training labels (which requires a sliding window setup),
    # we will just simulate the inference/prediction phase features.

    features_df = feature_gen.generate_features(
        candidates_df=candidates_df,
        history_df=train_df,
        target_df=None,  # No labels needed for inference
        cache_key="demo_inference",
        load_cached_data=False,
    )

    assert (
        "visual_consistency" in features_df.columns
    ), "Visual consistency feature missing"
    assert "global_pop" in features_df.columns, "Popularity feature missing"

    # 6. Ranking Model (Training & Inference)
    print("\n--- Step 6: Ranking Model ---")
    ranker = ranking_model.Ranker()

    # To demonstrate training, we need a labeled dataset.
    # We will split our 'features_df' artificially for this demo or generate a small training set.
    # Let's generate a proper small training set using the Validation Split we created earlier.

    # A. Create Training Candidates (History=Train, Target=Val)
    print("Generating training data for Ranker...")
    train_user_vecs = retriever.build_user_vectors(
        train_df, val_df["customer_id"].unique()
    )
    train_candidates = retriever.retrieve(
        train_user_vecs, seq_graph, visual_graph, val_df["customer_id"].unique()
    )

    if not train_candidates.empty:
        train_features = feature_gen.generate_features(
            candidates_df=train_candidates,
            history_df=train_df,
            target_df=val_df,  # Provides labels (1 if purchased in val, 0 otherwise)
            cache_key="demo_train",
            load_cached_data=False,
        )

        # Split into train/val for LightGBM
        # Simple random split for demo
        mask = np.random.rand(len(train_features)) < 0.8
        lgbm_train = train_features[mask]
        lgbm_val = train_features[~mask]

        # Train
        if len(lgbm_train) > 0 and len(lgbm_val) > 0:
            ranker.train(lgbm_train, lgbm_val)
        else:
            print("Warning: Not enough data for training split, skipping training.")
    else:
        print("Warning: No candidates retrieved for validation set.")

    # B. Inference on Test Set
    # We assume the model is trained (or we skip if data was insufficient, but assertions check flow)
    if ranker.model is not None:
        print("Predicting on test set...")
        scores = ranker.predict(features_df)
        features_df["score"] = scores

        # 7. Submission Generation
        print("\n--- Step 7: Submission ---")
        # We use the library function to format and save
        # We need to pass the sample submission dataframe format
        sample_sub = pd.read_parquet(config.TEST_METADATA)
        ranker.generate_and_save_submission(features_df, sample_sub)

        assert config.SUBMISSION_CSV.exists(), "Submission file was not created"

        # Verify submission content
        sub_df = pd.read_csv(config.SUBMISSION_CSV)
        print("Submission Head:")
        print(sub_df.head())
        assert len(sub_df) == len(target_customers), "Submission count mismatch"
        assert "prediction" in sub_df.columns, "Prediction column missing"

    print("\n=== Demo Execution Completed Successfully ===")
