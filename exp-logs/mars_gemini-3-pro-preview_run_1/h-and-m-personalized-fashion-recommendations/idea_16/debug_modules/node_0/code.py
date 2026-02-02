import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
import random
import shutil

# Import library modules
from library import settings
from library import metrics
from library.data_manager import TransactionLoader
from library.graph_model import InteractionGraph
from library.predictor import StratifiedRecommender


# ==========================================
# 1. Setup & Configuration
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_demo_environment():
    """
    Creates a temporary directory for demo data and overrides
    global settings to point to this directory.
    """
    # Define paths
    demo_dir = "./working/demo_env"
    demo_input = os.path.join(demo_dir, "input")
    demo_working = os.path.join(demo_dir, "working")

    # Clean up previous runs
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_input, exist_ok=True)
    os.makedirs(demo_working, exist_ok=True)

    # --- Create Subsampled Data ---
    print("Creating subsampled datasets for demonstration...")

    # Load a small fraction of the real metadata to ensure valid IDs and formats
    # We read the validation set because it's smaller, and use it to source users
    df_val_source = pd.read_csv(settings.PATH_VAL, nrows=50000)

    # Pick 100 unique customers to form our "Universe"
    sample_users = df_val_source["customer_id"].unique()[:100]

    # Filter transactions for these users
    df_sample = df_val_source[df_val_source["customer_id"].isin(sample_users)].copy()

    # Split this sample into "Train" and "Val" for the demo
    # We'll just split by time arbitrarily for the demo file creation
    # In reality, the library handles time splitting, but we need separate files
    # to mimic the metadata/train.csv and metadata/val.csv structure.

    # Let's pretend the first 80% of rows are train, last 20% are val
    split_idx = int(len(df_sample) * 0.8)
    demo_train_df = df_sample.iloc[:split_idx].copy()
    demo_val_df = df_sample.iloc[split_idx:].copy()

    # Create a sample submission file for these users
    demo_test_df = pd.DataFrame({"customer_id": sample_users})

    # Save to demo input directory
    path_train = os.path.join(demo_input, "train.csv")
    path_val = os.path.join(demo_input, "val.csv")
    path_test = os.path.join(demo_input, "test.csv")
    path_sub = os.path.join(demo_dir, "submission.csv")

    demo_train_df.to_csv(path_train, index=False)
    demo_val_df.to_csv(path_val, index=False)
    demo_test_df.to_csv(path_test, index=False)

    print(f"Sampled Train Rows: {len(demo_train_df)}")
    print(f"Sampled Val Rows: {len(demo_val_df)}")
    print(f"Sampled Test Users: {len(demo_test_df)}")

    # --- Monkey Patch Settings ---
    # This redirects the library to use our small files instead of the huge ones
    settings.WORKING_DIR = demo_working
    settings.PATH_TRAIN = path_train
    settings.PATH_VAL = path_val
    settings.PATH_TEST = path_test
    settings.PATH_SUBMISSION = path_sub

    # Update cache paths in settings to point to new working dir
    settings.CACHE_INTERACTION_MATRIX = os.path.join(
        demo_working, "interaction_matrix.npz"
    )
    settings.CACHE_SIMILARITY_MATRIX = os.path.join(
        demo_working, "similarity_matrix.npz"
    )
    settings.CACHE_USER_HISTORY = os.path.join(demo_working, "user_history.parquet")
    settings.CACHE_GLOBAL_TRENDS = os.path.join(demo_working, "global_trends.npy")
    settings.CACHE_ITEM_MAP = os.path.join(demo_working, "item_map.parquet")
    settings.CACHE_USER_MAP = os.path.join(demo_working, "user_map.parquet")

    # Adjust hyperparameters for speed
    settings.TOP_K_SIMILAR = 10  # Reduced from 100
    settings.TRAIN_WEEKS = (
        100  # Large window to ensure our small sample isn't filtered out by date
    )

    return demo_train_df, demo_val_df


# ==========================================
# 2. Metric Verification
# ==========================================
def verify_metrics():
    print("\n[Demo] Verifying Metrics...")

    # Test Case:
    # Actual: [1, 2, 3]
    # Predicted: [1, 4, 5] (1 Hit at rank 1, 2 Misses)
    # AP@3:
    # k=1: Hit (1). Precision=1/1. Score=1.
    # k=2: Miss (4).
    # k=3: Miss (5).
    # AP = 1.0 / min(3, 3) = 0.3333...

    actual = [1, 2, 3]
    predicted = [1, 4, 5]
    score = metrics.apk(actual, predicted, k=3)

    print(f"  APK([1,2,3], [1,4,5]) = {score:.4f}")
    assert np.isclose(
        score, 1.0 / 3.0
    ), f"APK calculation incorrect. Expected 0.333, got {score}"

    # Test MAP@12 Wrapper
    # Create dummy dataframes
    val_df = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "article_id": [101, 202],  # u1 bought 101, u2 bought 202
        }
    )

    # u1 predicts 101 (Perfect), u2 predicts 303 (Fail)
    sub_df = pd.DataFrame(
        {"customer_id": ["u1", "u2"], "prediction": ["101 999", "303 999"]}
    )

    # MAP = (1.0 + 0.0) / 2 = 0.5
    map_score = metrics.calculate_map_at_12(val_df, sub_df)
    print(f"  MAP@12 (Dummy) = {map_score:.4f}")
    assert np.isclose(
        map_score, 0.5
    ), f"MAP calculation incorrect. Expected 0.5, got {map_score}"
    print("  Metrics verified.")


# ==========================================
# 3. Data Manager Verification
# ==========================================
def verify_data_manager():
    print("\n[Demo] Verifying TransactionLoader...")
    loader = TransactionLoader()

    # Run in validation mode (splits data)
    # force load_cached_data=False to ensure logic runs
    train_df, val_df, user_map, item_map = loader.get_data(
        validation=True, load_cached_data=False
    )

    print(f"  Processed Train Shape: {train_df.shape}")
    print(f"  Processed Val Shape: {val_df.shape}")
    print(f"  User Map Size: {len(user_map)}")
    print(f"  Item Map Size: {len(item_map)}")

    # Assertions
    assert "weight" in train_df.columns, "Train data missing 'weight' column"
    assert "days_elapsed" in train_df.columns, "Train data missing 'days_elapsed'"
    assert not train_df.isnull().any().any(), "Train data contains NaNs"

    # Check if weights are decayed properly (should be <= 1.0)
    assert train_df["weight"].max() <= 1.000001, "Weights exceed 1.0"

    return train_df, user_map, item_map


# ==========================================
# 4. Graph Model Verification
# ==========================================
def verify_graph_model(train_df, user_map, item_map):
    print("\n[Demo] Verifying InteractionGraph...")
    n_users = len(user_map)
    n_items = len(item_map)

    graph = InteractionGraph(n_users, n_items)

    # Build graph from scratch
    graph.build(train_df, load_cached_data=False)

    X, S = graph.get_matrices()

    print(f"  Interaction Matrix X: {X.shape} (Stored as {type(X)})")
    print(f"  Similarity Matrix S: {S.shape} (Stored as {type(S)})")

    # Assertions
    assert X.shape == (n_users, n_items), "X shape mismatch"
    assert S.shape == (n_items, n_items), "S shape mismatch"
    assert sp.issparse(X), "X is not sparse"
    assert sp.issparse(S), "S is not sparse"

    # Check pruning (should not have more than TOP_K_SIMILAR items per row)
    # Note: S is symmetric-ish, but stored CSR.
    row_counts = np.diff(S.indptr)
    max_neighbors = row_counts.max()
    print(f"  Max neighbors in S: {max_neighbors}")
    assert (
        max_neighbors <= settings.TOP_K_SIMILAR
    ), f"Pruning failed. Max neighbors {max_neighbors} > {settings.TOP_K_SIMILAR}"


# ==========================================
# 5. Predictor Verification
# ==========================================
def verify_predictor():
    print("\n[Demo] Verifying StratifiedRecommender...")

    rec = StratifiedRecommender()

    # 1. Validation Run
    # This calculates MAP on the validation set we created
    print("  Running Validation Pipeline...")
    val_score = rec.run(validation=True)

    assert isinstance(val_score, float), "Validation run did not return a float score"
    print(f"  Pipeline Validation MAP: {val_score:.4f}")

    # 2. Submission Run
    # This generates the submission file
    print("  Running Submission Pipeline...")
    rec.run(validation=False)

    assert os.path.exists(settings.PATH_SUBMISSION), "Submission file not created"

    # Check submission content
    sub_df = pd.read_csv(settings.PATH_SUBMISSION)
    print(f"  Submission Rows: {len(sub_df)}")
    print(f"  Sample Prediction: {sub_df.iloc[0]['prediction']}")

    # Assert format
    assert "customer_id" in sub_df.columns
    assert "prediction" in sub_df.columns
    # Check prediction count (should be <= 12 items)
    pred_items = sub_df.iloc[0]["prediction"].split()
    assert len(pred_items) <= 12, "Predicted more than 12 items"

    print("  Predictor verified.")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    set_seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Metrics
        verify_metrics()

        # 3. Data Manager
        # We capture the outputs to pass to the graph verifier
        train_df, user_map, item_map = verify_data_manager()

        # 4. Graph Model
        verify_graph_model(train_df, user_map, item_map)

        # 5. Predictor
        # Note: Predictor re-loads data internally, but relies on the files we set up
        verify_predictor()

        print(
            "\n[Success] All TWIG-SR components demonstrated and verified successfully."
        )

    except AssertionError as e:
        print(f"\n[Failure] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
