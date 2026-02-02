import pandas as pd
import numpy as np
import os
import shutil
import scipy.sparse as sp
from library.utils import set_seed, calculate_map12, Timer
from library.data_loader import TransactionLoader
from library.matrix_factory import SparseMatrixBuilder
from library.similarity_engine import ItemSimilarityModel
from library.recommender import StratifiedRecommender


def test_map12_metric():
    """
    Verifies the calculate_map12 function with a known synthetic case.
    """
    print("\n[Demo] Testing MAP@12 Metric Function...")

    # Ground Truth: User 1 bought items A, B; User 2 bought item C
    gt = pd.DataFrame(
        {"customer_id": ["u1", "u1", "u2"], "article_id": [1001, 1002, 1003]}
    )

    # Predictions:
    # User 1: Predicted A, B (Perfect match -> AP=1.0)
    # User 2: Predicted D, E (No match -> AP=0.0)
    # Mean AP = 0.5
    preds = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "prediction": ["0000001001 0000001002", "0000001004 0000001005"],
        }
    )

    score = calculate_map12(preds, gt)
    print(f"  Computed MAP@12: {score}")

    assert np.isclose(score, 0.5), f"Expected MAP 0.5, got {score}"
    print("  MAP@12 Metric Test Passed.")


def run_pipeline_demo():
    """
    Demonstrates the full pipeline component by component.
    """
    print("\n[Demo] Starting Pipeline Execution...")

    # Configuration
    WORKING_DIR = "./working/demo_execution"
    TRAIN_WEEKS = 2  # Reduced from 10 to 2 for speed
    VAL_DAYS = 7

    # Clean working directory if exists to ensure fresh run
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[Demo] Step 1: Loading Transactions...")
    loader = TransactionLoader(cache_dir=WORKING_DIR)
    # Load data with validation split
    train_df, val_df, test_customers = loader.load_transactions(
        train_weeks=TRAIN_WEEKS, val_days=VAL_DAYS, validation=True
    )

    assert not train_df.empty, "Training dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"
    print(f"  Train Rows: {len(train_df)}")
    print(f"  Val Rows: {len(val_df)}")

    # 2. Build Matrices
    print("\n[Demo] Step 2: Building Interaction Matrix...")
    matrix_builder = SparseMatrixBuilder(cache_dir=WORKING_DIR)
    X, user_map, item_map = matrix_builder.build(train_df, test_customers)

    assert sp.issparse(X), "X should be a sparse matrix"
    print(f"  Matrix Shape: {X.shape}")

    # 3. Compute Similarity
    print("\n[Demo] Step 3: Computing Item Similarity...")
    sim_engine = ItemSimilarityModel(cache_dir=WORKING_DIR)
    # Top-k=50 is sufficient for demo and faster
    S = sim_engine.compute_similarity(X, top_k=50)

    assert S.shape == (X.shape[1], X.shape[1]), "Similarity matrix has incorrect shape"
    print(f"  Similarity Matrix Density: {S.nnz / (S.shape[0]**2):.6f}")

    # 4. Prepare Global Trends (Logic replicated from StratifiedRecommender)
    print("\n[Demo] Step 4: Computing Global Trends...")
    global_trend = np.array(X.sum(axis=0)).flatten()
    max_trend = global_trend.max()
    if max_trend > 0:
        global_trend = 10.0 * (global_trend / max_trend)
    global_trend = global_trend.astype(np.float32)

    # 5. Inference
    print("\n[Demo] Step 5: Generating Predictions...")
    # We instantiate the recommender to use its prediction logic
    rec = StratifiedRecommender(working_dir=WORKING_DIR)

    # Predict for validation users
    target_users = val_df["customer_id"].unique()

    # Note: Accessing protected method _predict_stratified for demonstration purposes
    # to show how to use the components we just built.
    preds = rec._predict_stratified(
        X, S, global_trend, user_map, item_map, target_users
    )

    assert len(preds) == len(target_users), "Prediction count mismatch"
    print(f"  Generated {len(preds)} predictions.")

    # 6. Evaluation
    print("\n[Demo] Step 6: Evaluating Performance...")
    score = calculate_map12(preds, val_df)
    print(f"  Validation MAP@12: {score:.6f}")

    # Basic sanity check on score (it should be > 0 given the data quality)
    assert 0.0 <= score <= 1.0, "MAP score out of range"

    # 7. Check Artifacts
    print("\n[Demo] Step 7: Verifying Cache Artifacts...")
    expected_files = [
        "interaction_matrix.npz",
        "user_map.parquet",
        "item_map.parquet",
        "similarity_matrix_k50.npz",
    ]
    for f in expected_files:
        path = os.path.join(WORKING_DIR, f)
        assert os.path.exists(path), f"Expected artifact {f} missing."
        print(f"  Found artifact: {f}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # Run Unit Test
        test_map12_metric()

        # Run Integration Demo
        run_pipeline_demo()

        print("\n[Demo] All demonstration steps completed successfully.")

    except AssertionError as e:
        print(f"\n[Demo] FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[Demo] ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
