import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing SMMC Pipeline Demo...")

    # =========================================================================
    # 1. SETUP & DATA SUBSETTING
    # =========================================================================
    # We create a mini dataset in the working directory to ensure the demo
    # runs quickly (within minutes) instead of hours.

    DEMO_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Creating mini-datasets for demonstration...")

    # Read a chunk of the real training data
    # We use metadata/train.csv as the source
    df_source = pd.read_csv("./metadata/train.csv", nrows=50000)

    # Select 100 unique customers to act as our "Test" set
    test_customers = df_source["customer_id"].unique()[:100]
    df_test_mini = pd.DataFrame({"customer_id": test_customers})

    # Filter transactions to include history for these customers
    # plus some extra transactions to ensure we have a pool of items
    df_train_mini = df_source[df_source["customer_id"].isin(test_customers)].copy()

    # Save these mini files
    mini_train_path = os.path.join(DEMO_DIR, "train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val.csv")  # Reusing train as val for demo
    mini_test_path = os.path.join(DEMO_DIR, "test.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_train_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    print(f"Mini-datasets saved to {DEMO_DIR}")

    # =========================================================================
    # 2. CONFIGURATION OVERRIDE
    # =========================================================================
    # We must patch the configuration paths BEFORE importing other library modules
    # so they pick up the paths to our mini-dataset.

    import library.config as config

    # Patch Input Paths
    config.TRAIN_PATH = mini_train_path
    config.VAL_PATH = mini_val_path
    config.TEST_PATH = mini_test_path

    # Patch Output/Cache Paths
    config.WORKING_DIR = CACHE_DIR
    config.SUBMISSION_DIR = DEMO_DIR

    # Update derived paths in config
    config.TRANSACTIONS_CACHE_PATH = os.path.join(
        CACHE_DIR, "transactions_processed.parquet"
    )
    config.ITEM_MAP_PATH = os.path.join(CACHE_DIR, "item_id_map.parquet")
    config.USER_HISTORY_PATH = os.path.join(CACHE_DIR, "user_history.parquet")
    config.BEHAVIOR_MATRIX_PATH = os.path.join(CACHE_DIR, "behavior_similarity.npz")
    config.VISUAL_MATRIX_PATH = os.path.join(CACHE_DIR, "visual_similarity.npz")
    config.VISUAL_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "visual_embeddings.npy")
    config.GLOBAL_TRENDS_PATH = os.path.join(CACHE_DIR, "global_trends.parquet")
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Reduce compute intensity for demo
    config.BATCH_SIZE = 32  # Smaller batch size for image embedding
    config.TOP_K_VISUAL = 10  # Fewer neighbors
    config.TOP_K_BEHAVIOR = 10

    # =========================================================================
    # 3. PIPELINE EXECUTION
    # =========================================================================

    # Now we import the modules, which will use the patched config
    from library.data_loader import TransactionLoader, IndexMapper, UserHistoryBuilder
    from library.visual_engine import ImageEmbedder, VisualSimilarityBuilder
    from library.behavioral_engine import CooccurrenceBuilder
    from library.smmc_recommender import SMMCModel
    from library.utils import apk

    # --- Step A: Verify Metric Logic ---
    print("\n[Step A] Verifying Metric Logic (APK)...")
    # Test Case:
    # Actual: [1, 2, 3]
    # Predicted: [1, 4, 2] -> 1 is correct (rank 1), 4 is wrong, 2 is correct (rank 3)
    # Precision@1: 1/1. Precision@2: 1/2. Precision@3: 2/3.
    # AP = (1/1 + 2/3) / 3 = (1 + 0.666) / 3 = 0.555...
    actual = ["A", "B", "C"]
    predicted = ["A", "D", "B"]
    score = apk(actual, predicted, k=3)
    expected = (1.0 + 2.0 / 3.0) / 3.0
    assert (
        abs(score - expected) < 1e-6
    ), f"APK calculation failed. Got {score}, expected {expected}"
    print("APK logic verified.")

    # --- Step B: Data Loading & Processing ---
    print("\n[Step B] Loading and Processing Transactions...")
    loader = TransactionLoader()
    # Force reload to use our mini dataset (ignore any existing cache from previous runs)
    transactions_df = loader.load_transactions(load_cached_data=False)

    assert not transactions_df.empty, "Transactions DataFrame is empty!"
    assert "days_elapsed" in transactions_df.columns, "days_elapsed column missing"
    print(f"Loaded {len(transactions_df)} transactions.")

    # --- Step C: Index Mapping ---
    print("\n[Step C] Fitting Index Mapper...")
    # Load test set for mapping
    test_df = pd.read_csv(config.TEST_PATH)
    mapper = IndexMapper()
    mapper.fit(transactions_df, test_df)

    n_users = mapper.get_num_users()
    n_items = mapper.get_num_items()
    print(f"Mapped {n_users} users and {n_items} items.")

    assert n_users == 100, f"Expected 100 users, got {n_users}"
    assert n_items > 0, "No items mapped."

    # --- Step D: User History Construction ---
    print("\n[Step D] Building User History Matrix...")
    history_builder = UserHistoryBuilder()
    user_history = history_builder.build_history(
        transactions_df, mapper, load_cached_data=False
    )

    assert user_history.shape == (
        n_users,
        n_items,
    ), "User history matrix shape mismatch"
    print(f"User History Matrix built: {user_history.shape}, NNZ: {user_history.nnz}")

    # --- Step E: Visual Engine ---
    print("\n[Step E] Running Visual Engine...")
    embedder = ImageEmbedder()
    # This will process images found in ./input/images corresponding to articles in our mini-set
    embeddings = embedder.extract_embeddings(mapper, load_cached_data=False)

    assert embeddings.shape == (
        n_items,
        2048,
    ), f"Embedding shape mismatch. Expected ({n_items}, 2048)"

    vis_builder = VisualSimilarityBuilder()
    visual_matrix = vis_builder.build_similarity_matrix(
        embeddings, load_cached_data=False
    )

    assert visual_matrix.shape == (n_items, n_items), "Visual matrix shape mismatch"
    print(
        f"Visual Similarity Matrix built: {visual_matrix.shape}, NNZ: {visual_matrix.nnz}"
    )

    # --- Step F: Behavioral Engine ---
    print("\n[Step F] Running Behavioral Engine...")
    beh_builder = CooccurrenceBuilder()
    behavior_matrix = beh_builder.build_similarity_matrix(
        user_history, load_cached_data=False
    )

    assert behavior_matrix.shape == (n_items, n_items), "Behavior matrix shape mismatch"
    print(
        f"Behavioral Similarity Matrix built: {behavior_matrix.shape}, NNZ: {behavior_matrix.nnz}"
    )

    # --- Step G: SMMC Inference ---
    print("\n[Step G] Running SMMC Model Inference...")
    model = SMMCModel(batch_size=50)  # Small batch size for demo

    submission_df = model.predict(
        user_history,
        behavior_matrix,
        visual_matrix,
        mapper,
        transactions_df,
        load_cached_data=False,
    )

    # --- Step H: Final Validation ---
    print("\n[Step H] Validating Submission...")
    assert (
        len(submission_df) == n_users
    ), f"Submission length {len(submission_df)} != Users {n_users}"
    assert "customer_id" in submission_df.columns
    assert "prediction" in submission_df.columns

    # Check format of first prediction
    sample_pred = submission_df.iloc[0]["prediction"]
    pred_items = sample_pred.split()
    assert len(pred_items) <= 12, "More than 12 predictions found for a user"

    # Check if file exists
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not saved."

    print(f"Submission successfully saved to {config.SUBMISSION_PATH}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set global seeds for reproducibility
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()
