import os
import shutil
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from pathlib import Path
from scipy import sparse

# Import provided library modules
import library.config
import library.data_utils
import library.visual_encoder
import library.retrieval_engine
import library.feature_generator
import library.ranker_model

from library.config import Config

# ==========================================
# 0. Setup and Monkey Patching for Demo Speed
# ==========================================


def setup_demo_env():
    """
    Configures the environment for a fast demonstration run.
    - Redirects working directories.
    - Reduces hyperparameters.
    - Monkey-patches data loading to use a small subset.
    """
    print("Setting up demonstration environment...")

    # 1. Set Random Seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # 2. Override Config Paths and Params
    # Use a separate directory for demo artifacts
    demo_working_dir = Path("./working/demo_execution")
    if demo_working_dir.exists():
        shutil.rmtree(demo_working_dir)
    demo_working_dir.mkdir(parents=True, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_ARTICLE_ID_MAP = demo_working_dir / "article_id_map.npy"
    Config.CACHE_CUSTOMER_ID_MAP = demo_working_dir / "customer_id_map.npy"
    Config.CACHE_ARTICLE_EMBEDDINGS = demo_working_dir / "image_embeddings.npy"
    Config.CACHE_VISUAL_GRAPH = demo_working_dir / "visual_graph.npz"
    Config.CACHE_SEQUENTIAL_GRAPH = demo_working_dir / "sequential_graph.npz"
    Config.CACHE_RANKER_TRAIN = demo_working_dir / "ranker_train.parquet"
    Config.CACHE_RANKER_VAL = demo_working_dir / "ranker_val.parquet"

    # Reduce computational load
    Config.RETRIEVAL_TOP_K = 20
    Config.VISUAL_KNN_K = 5
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["verbose"] = -1
    Config.NUM_WORKERS = 4

    # 3. Monkey Patch load_dataset to return a subset
    original_load_dataset = library.data_utils.load_dataset

    def mock_load_dataset():
        print(" [Mock] Loading and sampling dataset for demo...")
        # Load full data (fast read)
        train_df, val_df, test_df, articles_df, customers_df = original_load_dataset()

        # Sample Customers (Top active in train AND val to ensure coverage)
        # We need both because train/val are disjoint by user. Cite debug_lesson_3.
        top_train = train_df["customer_id"].value_counts().head(160).index
        top_val = val_df["customer_id"].value_counts().head(40).index
        top_customers = np.concatenate([top_train.values, top_val.values])

        # Filter Dataframes
        train_subset = train_df[train_df["customer_id"].isin(top_customers)].copy()
        val_subset = val_df[val_df["customer_id"].isin(top_customers)].copy()

        # For test, we take the intersection with our sampled customers + some randoms
        # to simulate cold start or unseen users if needed, but for this demo we stick to knowns
        test_subset = test_df[test_df["customer_id"].isin(top_customers)].copy()

        # Filter Articles (Only those referenced + some extras)
        active_articles = set(train_subset["article_id"].unique()) | set(
            val_subset["article_id"].unique()
        )
        articles_subset = articles_df[
            articles_df["article_id"].isin(active_articles)
        ].copy()

        # Filter Customers Metadata
        customers_subset = customers_df[
            customers_df["customer_id"].isin(top_customers)
        ].copy()

        print(
            f" [Mock] Subset Stats: Train={len(train_subset)}, Val={len(val_subset)}, "
            f"Test={len(test_subset)}, Articles={len(articles_subset)}"
        )

        return train_subset, val_subset, test_subset, articles_subset, customers_subset

    # Apply patch
    library.data_utils.load_dataset = mock_load_dataset
    library.feature_generator.load_dataset = mock_load_dataset


# ==========================================
# Main Execution Flow
# ==========================================

if __name__ == "__main__":
    setup_demo_env()

    print("\n" + "=" * 40)
    print("STEP 1: ID MAPPING & PREPARATION")
    print("=" * 40)

    # Force re-creation of maps with load_cached_data=False
    cust_to_idx, art_to_idx, cust_map, art_map = library.data_utils.get_id_maps(
        load_cached_data=False
    )

    # Verification
    assert len(cust_map) > 0, "Customer map is empty"
    assert len(art_map) > 0, "Article map is empty"
    assert Config.CACHE_ARTICLE_ID_MAP.exists(), "Article ID map file not saved"
    print(f"Mapped {len(cust_map)} customers and {len(art_map)} articles.")

    print("\n" + "=" * 40)
    print("STEP 2: VISUAL ENCODER & GRAPH")
    print("=" * 40)

    # 2.1 Generate Embeddings
    # We force scratch generation. This will use the mocked dataset's article list.
    # Note: The ImageEmbedder uses the Config.INPUT_DIR which is read-only.
    # It will try to find images for the subset of articles.
    embedder = library.visual_encoder.ImageEmbedder()
    embeddings = embedder.generate_embeddings(load_cached_data=False)

    assert embeddings.shape == (
        len(art_map),
        512,
    ), f"Embedding shape mismatch. Expected ({len(art_map)}, 512), got {embeddings.shape}"
    assert Config.CACHE_ARTICLE_EMBEDDINGS.exists(), "Embeddings file not saved"

    # 2.2 Build Visual Graph
    vis_graph = library.visual_encoder.build_visual_graph(load_cached_data=False)

    assert sparse.issparse(vis_graph), "Visual graph is not a sparse matrix"
    assert vis_graph.shape == (
        len(art_map),
        len(art_map),
    ), "Visual graph shape mismatch"
    print("Visual Graph constructed successfully.")

    print("\n" + "=" * 40)
    print("STEP 3: RETRIEVAL ENGINE")
    print("=" * 40)

    retriever = library.retrieval_engine.DualGraphRetriever()

    # Load data for retrieval context
    train_df, val_df, test_df, _, _ = library.data_utils.load_dataset()

    # Test generation on a small batch of customers
    target_customers = test_df["customer_id"].unique()[:10]
    candidates_df = retriever.generate_candidates(
        train_df, target_customers, load_cached_graphs=True
    )

    # Verification
    expected_cols = [
        "customer_id",
        "article_id",
        "score_seq",
        "score_vis",
        "score_hist",
        "rank",
    ]
    for col in expected_cols:
        assert col in candidates_df.columns, f"Missing column {col} in candidates"

    assert candidates_df["customer_id"].nunique() == len(
        target_customers
    ), "Not all target customers received candidates"
    print(
        f"Generated {len(candidates_df)} candidates for {len(target_customers)} customers."
    )

    print("\n" + "=" * 40)
    print("STEP 4: FEATURE GENERATION")
    print("=" * 40)

    feature_factory = library.feature_generator.RankerFeatureFactory()

    # 4.1 Create Train Set
    # This uses sliding windows on the mocked data
    ranker_train = feature_factory.create_train_dataset(load_cached_data=False)

    assert not ranker_train.empty, "Ranker training set is empty"
    assert "label" in ranker_train.columns, "Label column missing in training set"
    assert (
        "visual_consistency" in ranker_train.columns
    ), "Feature 'visual_consistency' missing"

    # 4.2 Create Validation Set
    ranker_val = feature_factory.create_validation_dataset(load_cached_data=False)
    assert not ranker_val.empty, "Ranker validation set is empty"

    print(
        f"Training Samples: {len(ranker_train)}, Validation Samples: {len(ranker_val)}"
    )

    print("\n" + "=" * 40)
    print("STEP 5: RANKER TRAINING")
    print("=" * 40)

    ranker = library.ranker_model.LGBMRanker()
    ranker.train(ranker_train, ranker_val)

    assert (Config.WORKING_DIR / "lgbm_model.txt").exists(), "Model file not saved"
    print("Model trained and saved.")

    print("\n" + "=" * 40)
    print("STEP 6: INFERENCE & SUBMISSION")
    print("=" * 40)

    # 6.1 Create Inference Set (Candidates for Test Users)
    ranker_inference = feature_factory.create_inference_dataset(load_cached_data=False)
    assert not ranker_inference.empty, "Inference dataset is empty"

    # 6.2 Predict and Submit
    ranker.predict(ranker_inference)

    assert Config.SUBMISSION_PATH.exists(), "Submission file not created"

    # Verify Submission Format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "customer_id" in sub_df.columns and "prediction" in sub_df.columns
    ), "Invalid submission columns"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check prediction format (string of space-separated IDs)
    sample_pred = sub_df.iloc[0]["prediction"]
    if pd.isna(sample_pred) or sample_pred == "":
        print(
            "Warning: First prediction is empty (Cold start fallback might be needed or no history for user)"
        )
    else:
        items = sample_pred.split(" ")
        assert len(items) <= 12, "More than 12 items predicted"
        # Check for 10-digit format
        assert len(items[0]) == 10, f"Invalid article ID format: {items[0]}"

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\nDemo execution completed successfully.")
