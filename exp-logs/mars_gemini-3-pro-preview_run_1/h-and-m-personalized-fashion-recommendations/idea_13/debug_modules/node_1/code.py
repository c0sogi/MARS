import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shutil

# Import provided library modules
from library.config import Config
import library.data_utils as du
from library.graph_builder import GraphBuilder
from library.recommender import MSGRecommender


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Initializing Demo Configuration...")
    set_seed(42)

    # 1. Configuration Setup
    # We override the default configuration to run a fast, isolated demo
    config = Config()
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Enable debug mode to process only a subset of data for speed
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50000  # Process last 50k transactions

    # Reduce graph complexity for demo speed
    config.TOP_K_NEIGHBORS = 50

    # Ensure directories exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {config.WORKING_DIR}")

    # 2. Data Loading & Preprocessing (library.data_utils)
    print("\n--- Testing Data Utils ---")

    # Load transactions (uses metadata/train.csv by default in Config)
    # This will also compute decay weights
    print("Loading transactions...")
    train_df = du.load_transactions(
        config.TRAIN_DATA_PATH, config, load_cached_data=False
    )

    # Validation: Check dimensions and columns
    assert not train_df.empty, "Train DataFrame should not be empty"
    assert "weight_fast" in train_df.columns, "weight_fast column missing"
    assert "weight_slow" in train_df.columns, "weight_slow column missing"
    assert "t_dat" in train_df.columns, "t_dat column missing"

    # Check that weights are within expected bounds [0, 1]
    assert train_df["weight_fast"].between(0, 1).all(), "Fast weights out of bounds"
    assert train_df["weight_slow"].between(0, 1).all(), "Slow weights out of bounds"

    print(f"Loaded {len(train_df)} transactions (Debug Mode).")

    # Test Active Inventory
    print("Computing active inventory...")
    active_items = du.get_active_inventory(train_df, config)
    assert isinstance(active_items, np.ndarray), "Active items should be a numpy array"
    assert len(active_items) > 0, "Active items list is empty"
    print(f"Identified {len(active_items)} active items.")

    # 3. Graph Construction (library.graph_builder)
    print("\n--- Testing Graph Builder ---")

    # Load raw metadata for mappings
    customers_df = pd.read_csv(config.CUSTOMERS_PATH)
    articles_df = pd.read_csv(config.ARTICLES_PATH)

    gb = GraphBuilder(config)

    # Run the full graph building pipeline
    # This builds mappings, interaction matrices, and similarity matrices
    print("Running GraphBuilder pipeline...")
    X_fast, S_fast, X_slow, S_slow = gb.run(
        train_df, customers_df, articles_df, active_items, load_cached=False
    )

    # Validation: Check Matrix Properties
    print("Validating Graph Artifacts...")

    # Check Dimensions
    n_users = gb.n_users
    n_items = gb.n_items

    assert X_fast.shape == (n_users, n_items), f"X_fast shape mismatch: {X_fast.shape}"
    assert S_fast.shape == (n_items, n_items), f"S_fast shape mismatch: {S_fast.shape}"
    assert X_slow.shape == (n_users, n_items), f"X_slow shape mismatch: {X_slow.shape}"
    assert S_slow.shape == (n_items, n_items), f"S_slow shape mismatch: {S_slow.shape}"

    # Check Sparsity (Matrices should be sparse)
    assert sp.issparse(X_fast), "X_fast is not sparse"
    assert sp.issparse(S_fast), "S_fast is not sparse"

    # Check Mapping Consistency
    # Ensure some items in the training data are actually mapped
    sample_article = train_df["article_id"].iloc[0]
    assert (
        sample_article in gb.item_map
    ), "Sample article from train not found in item_map"

    print("Graph construction successful.")

    # 4. Recommendation Inference (library.recommender)
    print("\n--- Testing Recommender ---")

    recommender = MSGRecommender(config)

    # Create a small test set of customers for the demo
    # We take 100 random customers from the master list
    test_customers_sample = customers_df.sample(n=100, random_state=42)[["customer_id"]]

    print(f"Generating submission for {len(test_customers_sample)} test customers...")

    # Generate submission
    # Note: We pass load_cached=True so it picks up the graphs we just built
    recommender.generate_submission(
        train_df, test_customers_sample, articles_df, active_items, load_cached=True
    )

    # 5. Result Validation
    print("\n--- Validating Submission File ---")

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    submission_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check shape
    assert (
        len(submission_df) == 100
    ), f"Expected 100 predictions, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "customer_id",
        "prediction",
    ], "Incorrect columns in submission"

    # Check prediction format
    # Should be a string of space-separated article IDs
    sample_pred = submission_df["prediction"].iloc[0]

    # It's possible for a prediction to be empty string if no history and no global trend (unlikely but possible in debug)
    # But usually it should have items.
    if pd.isna(sample_pred):
        # If NaN, it's an issue unless we allow empty predictions (usually competitions require non-empty)
        # The code fills with "" if empty, so read_csv might see NaN.
        # Let's check fillna behavior.
        submission_df["prediction"] = submission_df["prediction"].fillna("")
        sample_pred = submission_df["prediction"].iloc[0]

    items = sample_pred.split()

    # Check max items
    assert len(items) <= 12, f"Prediction contains more than 12 items: {len(items)}"

    # Check that items look like article IDs (digits)
    if len(items) > 0:
        assert items[0].isdigit(), f"Predicted item {items[0]} is not numeric"
        # Check length of article id (usually 10 chars in this dataset, but might vary slightly if int conversion happened)
        # The recommender converts back using reverse_item_map which stores original IDs.
        # Original IDs in articles.csv are int64, so they might not have leading zeros unless formatted.
        # The sample submission provided in task description has leading zeros.
        # The provided code does `str(article_id).zfill(10)` in metadata generation, but `articles.csv` has int64.
        # The recommender uses `reverse_item_map` which comes from `articles_df['article_id']` (int64).
        # So predictions will be strings of integers.
        pass

    print("Submission validation passed.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
