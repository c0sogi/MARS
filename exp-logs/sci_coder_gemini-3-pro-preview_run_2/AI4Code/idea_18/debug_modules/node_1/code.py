import os
import shutil
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import count_inversions, kendall_tau
from library.data_processing import NotebookProcessor
from library.feature_extraction import FeatureEngine
from library.modeling import Stage1Ridge, Stage2LGBM
from library.workflow import NotebookOrderingWorkflow


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Enable Debug mode to sample only 100 notebooks per split
    Config.DEBUG = True

    # Reduce dimensionality and complexity for speed
    Config.VOCAB_SIZE = 500
    Config.SVD_COMPONENTS = 10
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.NUM_WORKERS = 2  # Low worker count for stability in demo env

    # Clean working directory to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    Config.setup()
    Config.set_seeds()
    print("Configuration applied. Working directory cleaned.")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test count_inversions
    # [2, 0, 1] -> 2 comes before 0 (swap), 2 comes before 1 (swap). Total 2.
    inv_test = count_inversions([2, 0, 1])
    assert inv_test == 2, f"count_inversions failed: expected 2, got {inv_test}"

    # Test kendall_tau
    # Identical order -> Score 1.0
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    score_perfect = kendall_tau(df_gt, df_pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"kendall_tau perfect match failed: {score_perfect}"

    # Reverse order -> Score -1.0 (Worst case)
    # n=3, pairs=3. Swaps=3. K = 1 - 4*(3/6) = -1.
    df_pred_worst = pd.DataFrame({"id": ["nb1"], "cell_order": ["c b a"]})
    score_worst = kendall_tau(df_gt, df_pred_worst)
    assert np.isclose(
        score_worst, -1.0
    ), f"kendall_tau worst case failed: {score_worst}"

    print("Utility functions verified.")

    # --------------------------------------------------------------------------
    # 3. Data Processing
    # --------------------------------------------------------------------------
    print("\n[3] Testing Data Processing (NotebookProcessor)...")
    processor = NotebookProcessor()

    # Load Train Data (Force processing from metadata)
    df_train = processor.load_dataset(split="train", load_cached_data=False)
    assert not df_train.empty, "Training dataframe is empty."
    assert (
        "pct_rank" in df_train.columns
    ), "Target column 'pct_rank' missing in train data."
    print(f"Processed {len(df_train)} training cells (Debug Sample).")

    # Load Test Data
    df_test = processor.load_dataset(split="test", load_cached_data=False)
    assert not df_test.empty, "Test dataframe is empty."
    print(f"Processed {len(df_test)} test cells (Debug Sample).")

    # --------------------------------------------------------------------------
    # 4. Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[4] Testing Feature Extraction (FeatureEngine)...")
    engine = FeatureEngine()

    # Fit Global Vectorizers
    engine.fit_global_vectorizers(df_train, load_cached_models=False)
    assert os.path.exists(engine.tfidf_path), "TF-IDF model file not found."
    assert os.path.exists(engine.svd_path), "SVD model file not found."

    # Extract Stage 1 Features (Sparse TF-IDF)
    train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(drop=True)
    X_s1 = engine.get_stage1_features(train_md)
    assert X_s1.shape[0] == len(train_md), "Stage 1 feature row count mismatch."
    assert X_s1.shape[1] == Config.VOCAB_SIZE, "Stage 1 feature vocab size mismatch."
    print(f"Stage 1 features shape: {X_s1.shape}")

    # Extract Stage 2 Features (Multi-View Instance)
    # This processes the full dataframe (code+md) but returns features for md rows
    X_s2 = engine.get_stage2_features(df_train, split="train", load_cached_data=False)
    assert len(X_s2) == len(train_md), "Stage 2 feature row count mismatch."
    # Check for specific feature columns
    expected_col = "lexical_neighbor_1_score"
    assert (
        expected_col in X_s2.columns
    ), f"Expected column {expected_col} missing in Stage 2 features."
    print(f"Stage 2 features shape: {X_s2.shape}")

    # --------------------------------------------------------------------------
    # 5. Modeling
    # --------------------------------------------------------------------------
    print("\n[5] Testing Modeling Components...")

    y_train = train_md["pct_rank"].values

    # Stage 1: Ridge
    print("Training Stage 1 Ridge...")
    ridge_model = Stage1Ridge()
    ridge_model.fit(X_s1, y_train)
    assert os.path.exists(ridge_model.model_path), "Ridge model file was not saved."
    preds_s1 = ridge_model.predict(X_s1)
    assert len(preds_s1) == len(y_train), "Ridge predictions length mismatch."

    # Stage 2: LightGBM
    print("Training Stage 2 LightGBM...")
    lgbm_model = Stage2LGBM()
    # Prepare features (drop non-feature cols)
    drop_cols = ["id", "cell_id"]
    X_lgbm = X_s2.drop(columns=drop_cols, errors="ignore")

    lgbm_model.fit(X_lgbm, y_train)
    assert os.path.exists(lgbm_model.model_path), "LightGBM model file was not saved."
    preds_s2 = lgbm_model.predict(X_lgbm)
    assert len(preds_s2) == len(y_train), "LightGBM predictions length mismatch."

    # --------------------------------------------------------------------------
    # 6. Full Workflow Execution
    # --------------------------------------------------------------------------
    print("\n[6] Executing Full Workflow (Train + Inference)...")

    workflow = NotebookOrderingWorkflow()

    # Run Training Pipeline
    # This will use the cached data/models we just generated/verified where possible
    workflow.train()

    # Run Inference Pipeline
    workflow.predict()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert not df_sub.empty, "Submission file is empty."
    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect."

    print(f"Submission generated with {len(df_sub)} rows.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
