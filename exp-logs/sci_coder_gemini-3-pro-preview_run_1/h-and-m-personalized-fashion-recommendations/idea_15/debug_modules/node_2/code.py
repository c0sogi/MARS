import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shutil
import warnings
from datetime import datetime, timedelta

# Import provided library modules
from library import config, utils, data_handler, sparse_engine, adipc_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    # torch is not used in the provided library, but good practice if extended
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_demo_config():
    """
    Overrides default configuration to use a temporary directory and debug sampling
    to ensure the script completes quickly.
    """
    print(">>> Setting up demo configuration...")

    # Define a specific working directory for this execution
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config globals
    config.WORKING_DIR = demo_working_dir

    # Update cache paths based on new working dir
    config.CACHE_ITEM_MAP = "item_map.parquet"
    config.CACHE_USER_MAP = "user_map.parquet"
    config.CACHE_INTERACTION_MATRIX = "interaction_matrix.npz"
    config.CACHE_SIMILARITY_MATRIX = "similarity_matrix.npz"
    config.CACHE_GLOBAL_TREND = "global_trends.npy"
    config.CACHE_INVENTORY_MASK = "inventory_mask.npy"
    config.CACHE_USER_HISTORY = "user_history.parquet"

    # Set debug size to limit processing time
    # 50,000 transactions is enough to verify the pipeline works
    config.DEBUG_SAMPLE_SIZE = 50000

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print("-" * 30)


def test_utils():
    """
    Verifies the logic in library/utils.py
    """
    print(">>> Testing Utils...")

    # 1. Test Decay Weights
    ref_date = pd.Timestamp("2020-09-22")
    dates = pd.to_datetime(
        ["2020-09-22", "2020-09-21", "2020-09-20"]
    )  # 0, 1, 2 days elapsed
    decay_rate = 1.0

    weights = utils.calculate_decay_weights(dates, ref_date, decay_rate)

    # Expected: (1+0)^-1 = 1.0, (1+1)^-1 = 0.5, (1+2)^-1 = 0.333...
    expected = np.array([1.0, 0.5, 1 / 3], dtype=np.float32)

    assert np.allclose(
        weights, expected, atol=1e-5
    ), f"Decay weights mismatch. Got {weights}, expected {expected}"
    print("Utils: calculate_decay_weights passed.")

    # 2. Test AP@k
    actual = [1, 2, 3, 4, 5]
    predicted_perfect = [1, 2, 3, 4, 5]
    predicted_mixed = [1, 99, 2, 88, 3]  # Hits at rank 1, 3, 5

    score_perfect = utils.apk(actual, predicted_perfect, k=12)
    assert score_perfect == 1.0, f"Perfect AP should be 1.0, got {score_perfect}"

    # Calculation for mixed:
    # Rank 1: Hit (1). Precision=1/1.
    # Rank 2: Miss (99).
    # Rank 3: Hit (2). Precision=2/3.
    # Rank 4: Miss (88).
    # Rank 5: Hit (3). Precision=3/5.
    # Sum = 1 + 0.666 + 0.6 = 2.2666
    # Div by min(len(actual), 12) = 5 -> 2.2666 / 5 = 0.4533
    score_mixed = utils.apk(actual, predicted_mixed, k=12)
    assert (
        0.45 < score_mixed < 0.46
    ), f"Mixed AP calculation incorrect. Got {score_mixed}"
    print("Utils: apk passed.")

    # 3. Test MAP@12
    valid_df = pd.DataFrame({"customer_id": ["c1", "c2"], "article_id": [101, 202]})
    # c1 predicts correctly, c2 predicts wrongly
    sub_df = pd.DataFrame(
        {"customer_id": ["c1", "c2"], "prediction": ["101 102 103", "303 304 305"]}
    )

    # c1 score: 1.0 (hit at rank 1, total 1 ground truth)
    # c2 score: 0.0
    # Mean: 0.5
    map_score = utils.calculate_map12(valid_df, sub_df)
    assert map_score == 0.5, f"MAP calculation incorrect. Got {map_score}"
    print("Utils: calculate_map12 passed.")
    print("-" * 30)


def test_sparse_engine():
    """
    Verifies the logic in library/sparse_engine.py using synthetic data.
    """
    print(">>> Testing Sparse Engine...")
    engine = sparse_engine.SparseEngine()

    # Synthetic Data: 3 Users, 4 Items
    # User 0 bought Item 0
    # User 1 bought Item 0 and Item 1
    # User 2 bought Item 2 and Item 3
    df = pd.DataFrame({"user_idx": [0, 1, 1, 2, 2], "article_idx": [0, 0, 1, 2, 3]})
    weights = np.ones(len(df), dtype=np.float32)
    num_users = 3
    num_items = 4

    # 1. Build Matrix
    matrix = engine.build_user_item_matrix(df, weights, num_users, num_items)
    assert matrix.shape == (3, 4)
    assert matrix[1, 0] == 1.0
    print("SparseEngine: build_user_item_matrix passed.")

    # 2. IDF Weighting
    # Item 0 appears twice (User 0, User 1) -> DF=2
    # Item 1 appears once (User 1) -> DF=1
    # IDF(0) = log(3 / (2+1)) = log(1) = 0.0
    # IDF(1) = log(3 / (1+1)) = log(1.5) > 0
    weighted_matrix = engine.apply_idf_weighting(matrix)
    assert weighted_matrix[0, 0] == 0.0, "Item 0 should have 0 weight (log(1))"
    assert weighted_matrix[1, 1] > 0.0, "Item 1 should have positive weight"
    print("SparseEngine: apply_idf_weighting passed.")

    # 3. Normalize Rows
    # Create a simple matrix for normalization
    m_simple = sp.csr_matrix([[3.0, 4.0]], dtype=np.float32)  # Norm = 5
    m_norm = engine.normalize_rows(m_simple)
    assert np.isclose(m_norm[0, 0], 0.6), "Normalization failed (3/5)"
    assert np.isclose(m_norm[0, 1], 0.8), "Normalization failed (4/5)"
    print("SparseEngine: normalize_rows passed.")

    # 4. Item Similarity
    # Define matrix where Item 0 and Item 1 are co-purchased by User 0
    # User 0: [1, 1, 0]
    # User 1: [0, 0, 1]
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 2])
    data = np.array([1, 1, 1], dtype=np.float32)
    interaction = sp.csr_matrix((data, (rows, cols)), shape=(2, 3))
    interaction = engine.normalize_rows(
        interaction
    )  # User 0 vector becomes [0.707, 0.707, 0]

    sim = engine.compute_item_similarity(interaction, top_k=2)

    # Sim(0, 1) should be high (dot product of columns)
    # Sim(0, 0) should be 0 (diagonal zeroed)
    assert sim[0, 0] == 0.0, "Diagonal should be zero"
    assert sim[0, 1] > 0.0, "Similarity between co-purchased items should be > 0"
    assert sim[0, 2] == 0.0, "No overlap between Item 0 and Item 2"

    print("SparseEngine: compute_item_similarity passed.")
    print("-" * 30)


def run_pipeline_demo():
    """
    Runs the full ADIPC pipeline on a small subset of the real data.
    """
    print(">>> Running Full Pipeline Demo...")

    # Initialize Model
    model = adipc_model.ADIPCRecommender()

    # 1. Fit (Validation Mode)
    # This will trigger DataHandler to load data, sample 50k rows, and build matrices
    print("\n[Step 1] Fitting Model (Validation Mode)...")
    model.fit(mode="validation", load_cached_data=False)

    # Verify artifacts were created
    assert model.similarity_matrix is not None
    assert model.global_trend is not None
    assert model.inventory_mask is not None
    print("Model fitted successfully. Artifacts generated.")

    # 2. Load Validation Data for Prediction
    print("\n[Step 2] Loading Validation Context...")
    # We reload the dataset to get the target users and history split
    dataset = model.data_handler.load_dataset(mode="validation", load_cached_data=True)

    history_df = dataset["history_df"]
    future_df = dataset["future_df"]
    target_users = dataset["target_users"]
    cutoff_date = dataset["cutoff_date"]

    print(f"Target Users for Validation: {len(target_users)}")
    print(f"Ground Truth Rows: {len(future_df)}")

    if len(target_users) == 0:
        print(
            "Warning: No target users found in the debug sample. Skipping prediction."
        )
        return

    # 3. Predict
    print("\n[Step 3] Generating Predictions...")
    # Predict for a subset of target users to save time if there are many
    subset_users = target_users[:1000]
    submission_df = model.predict(subset_users, history_df, cutoff_date, batch_size=500)

    assert "customer_id" in submission_df.columns
    assert "prediction" in submission_df.columns
    assert len(submission_df) == len(subset_users)
    print(f"Generated predictions for {len(submission_df)} users.")
    print(f"Sample Prediction: \n{submission_df.head(1)}")

    # 4. Evaluate
    print("\n[Step 4] Evaluating MAP@12...")
    # Filter ground truth to the subset of users we predicted for
    subset_cust_ids = set(submission_df["customer_id"])
    relevant_future_df = future_df[
        future_df["user_idx"].isin(
            model.user_map[model.user_map["customer_id"].isin(subset_cust_ids)][
                "user_idx"
            ]
        )
    ]

    # We need to map future_df article_idx back to article_id for scoring
    # because calculate_map12 expects article_ids (integers)
    idx_to_article = model.item_map.set_index("article_idx")["article_id"].to_dict()
    idx_to_cust = model.user_map.set_index("user_idx")["customer_id"].to_dict()

    eval_df = relevant_future_df.copy()
    eval_df["article_id"] = eval_df["article_idx"].map(idx_to_article)
    eval_df["customer_id"] = eval_df["user_idx"].map(idx_to_cust)

    score = utils.calculate_map12(eval_df, submission_df)
    print(f"\n>>> Demo MAP@12 Score: {score:.5f}")

    # Sanity check: Score should be a number between 0 and 1
    assert 0.0 <= score <= 1.0
    print("-" * 30)


if __name__ == "__main__":
    set_seed(42)

    # 1. Setup
    setup_demo_config()

    # 2. Unit Tests
    test_utils()
    test_sparse_engine()

    # 3. Integration Test
    run_pipeline_demo()

    print("\n>>> All demonstrations completed successfully.")
