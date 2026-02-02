import sys
import os
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Suppress warnings and TQDM output for clean execution
warnings.filterwarnings("ignore")

# Patch tqdm to be silent before importing libraries that use it
import tqdm


def no_op_tqdm(iterable=None, *args, **kwargs):
    if iterable is None:
        return
    return iterable


tqdm.tqdm = no_op_tqdm

# 2. Import Library Components
from library.config import Config
from library.data import DataManager
from library.retrieval import HybridRetrieval
from library.features import FeatureEngineer
from library.ranker import Ranker
from library.evaluation import Evaluator


def set_seed(seed=42):
    np.random.seed(seed)
    # torch.manual_seed(seed) # Not used in this specific pipeline


def main():
    print("=== Starting Recommendation System Demo ===")

    # --- Step 1: Configuration ---
    # We modify the Config class attributes directly to suit a fast demo run.
    print("[Demo] Configuring parameters for speed...")

    # Use a separate working directory for this demo to avoid conflicts
    demo_working_dir = Path("./working/demo_run")
    if demo_working_dir.exists():
        shutil.rmtree(demo_working_dir)
    demo_working_dir.mkdir(parents=True, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir

    # Reduce Model Complexity
    Config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for speed
    Config.LGBM_PARAMS["min_data_in_leaf"] = 5
    Config.LGBM_PARAMS["num_leaves"] = 16
    Config.LGBM_PARAMS["verbose"] = -1

    # Reduce Retrieval Scope
    Config.RETRIEVAL_HISTORY_WEEKS = 2  # Only look at 2 weeks of history
    Config.TOP_K_COOC = 10  # Fewer candidates
    Config.TOP_K_REPURCHASE = 5
    Config.TOP_K_POPULARITY = 5

    # Update paths in Config that depend on WORKING_DIR
    Config.PATH_ARTICLE_MAP = Config.WORKING_DIR / "article_map.parquet"
    Config.PATH_CUSTOMER_MAP = Config.WORKING_DIR / "customer_map.parquet"
    Config.PATH_COOC_MATRIX = Config.WORKING_DIR / "cooccurrence_matrix.npz"
    Config.PATH_GLOBAL_POPULARITY = Config.WORKING_DIR / "global_popularity.npy"
    Config.PATH_CANDIDATES_TRAIN = Config.WORKING_DIR / "candidates_train.parquet"
    Config.PATH_CANDIDATES_TEST = Config.WORKING_DIR / "candidates_test.parquet"
    Config.PATH_SUBMISSION = Config.SUBMISSION_DIR / "demo_submission.csv"

    # --- Step 2: Data Loading ---
    print("[Demo] Loading and Preprocessing Data...")
    dm = DataManager()
    # We force reload=False (which means it will process from scratch because the demo dir is empty)
    # This reads the metadata parquets, maps IDs, and splits time.
    data = dm.load_data(load_cached_data=True)

    train_df = data["train"]
    val_df = data["val"]
    articles_df = data["articles"]
    customers_df = data["customers"]

    # --- Step 3: Subsampling for Demo ---
    print("[Demo] Subsampling data for rapid execution...")
    # We will select a small subset of customers from the validation set
    SAMPLE_SIZE = 1000
    unique_val_cust = val_df["customer_id_idx"].unique()

    if len(unique_val_cust) > SAMPLE_SIZE:
        sampled_cust_ids = np.random.choice(unique_val_cust, SAMPLE_SIZE, replace=False)
    else:
        sampled_cust_ids = unique_val_cust

    # Filter Validation Data
    val_df_small = val_df[val_df["customer_id_idx"].isin(sampled_cust_ids)].reset_index(
        drop=True
    )

    # Filter Training Data
    # We keep transactions for the sampled customers + a random sample of others to ensure
    # the co-occurrence matrix isn't empty/trivial, but small enough to be fast.
    target_history = train_df[train_df["customer_id_idx"].isin(sampled_cust_ids)]
    other_history = train_df.sample(n=50000, random_state=42)  # 50k random transactions
    train_df_small = (
        pd.concat([target_history, other_history])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"   Train Subset: {len(train_df_small)} rows")
    print(f"   Val Subset: {len(val_df_small)} rows")

    # --- Step 4: Retrieval ---
    print("[Demo] Stage 1: Candidate Retrieval...")
    retriever = HybridRetrieval()
    # Update instance working dir (since it might have initialized before Config update if not careful,
    # though here we instantiated after. Good practice to ensure.)
    retriever.working_dir = Config.WORKING_DIR

    # Generate candidates for the sampled validation customers
    # We treat 'val' as our target prediction set for this demo
    target_customers = val_df_small["customer_id_idx"].unique()

    candidates = retriever.generate_candidates(
        train_df_small, target_customers, mode="demo_val", load_cached_data=False
    )

    # Verification
    if candidates.empty:
        raise RuntimeError("Retrieval generated no candidates! Check data sampling.")

    expected_cols = [
        "customer_id_idx",
        "article_id_idx",
        "cooc_score",
        "repurchase_score",
        "pop_score",
    ]
    for col in expected_cols:
        assert col in candidates.columns, f"Missing candidate column: {col}"

    print(
        f"   Generated {len(candidates)} candidates for {len(target_customers)} customers."
    )

    # --- Step 5: Feature Engineering ---
    print("[Demo] Stage 2: Feature Engineering...")
    fe = FeatureEngineer()
    fe.working_dir = Config.WORKING_DIR

    # We use 'train' mode to generate labels based on val_df_small (our ground truth)
    features_df = fe.generate_features(
        candidates,
        train_df_small,
        articles_df,
        customers_df,
        mode="train",
        labeled_data=val_df_small,
        load_cached_data=False,
    )

    # Verification
    assert "label" in features_df.columns, "Labels were not generated."
    assert "product_type_no" in features_df.columns, "Item features missing."
    assert "age" in features_df.columns, "User features missing."

    # Check for interaction features
    affinity_col = f"{Config.AFFINITY_COLS[0]}_affinity"
    assert (
        affinity_col in features_df.columns
    ), f"Affinity feature {affinity_col} missing."

    print(f"   Feature Matrix Shape: {features_df.shape}")

    # --- Step 6: Ranking ---
    print("[Demo] Stage 3: Ranking (LightGBM)...")
    ranker = Ranker()
    ranker.working_dir = Config.WORKING_DIR

    # Identify feature columns (exclude IDs and Label)
    exclude_cols = ["customer_id_idx", "article_id_idx", "label"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Split the features_df into a training and validation set for the Ranker
    # In a real scenario, we'd have separate time periods. Here we split by customer.
    unique_custs = features_df["customer_id_idx"].unique()
    split_idx = int(len(unique_custs) * 0.8)
    train_custs = unique_custs[:split_idx]
    val_custs = unique_custs[split_idx:]

    rank_train = features_df[features_df["customer_id_idx"].isin(train_custs)]
    rank_val = features_df[features_df["customer_id_idx"].isin(val_custs)]

    # Train
    ranker.train(rank_train, rank_val, feature_cols)

    # Predict
    # We predict on the 'rank_val' set to evaluate performance
    print("[Demo] Generating Predictions...")
    submission = ranker.predict(rank_val.copy(), feature_cols)

    # Verification
    assert "customer_id" in submission.columns
    assert "prediction" in submission.columns
    assert len(submission) > 0

    # --- Step 7: Evaluation ---
    print("[Demo] Stage 4: Evaluation...")
    evaluator = Evaluator()

    # Filter ground truth to match the customers in the submission
    sub_cust_ids = submission["customer_id"].unique()
    relevant_ground_truth = val_df_small[val_df_small["customer_id"].isin(sub_cust_ids)]

    map12 = evaluator.calculate_map12(relevant_ground_truth, submission)

    assert 0.0 <= map12 <= 1.0, f"MAP@12 score {map12} is invalid."
    print(f"   Final MAP@12 Score on Demo Set: {map12:.6f}")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    set_seed(42)
    main()
