import os
import sys
import shutil
import numpy as np
import pandas as pd
from library.config import Config, setup_reproducibility
from library.utils import kendall_tau_metric, read_notebook
from library.data_loader import NotebookDataLoader
from library.feature_extraction import FeaturePipeline
from library.model_definitions import Level1Ridge, Level2GBM
from library.pipeline_manager import PipelineManager

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks
    Config.N_FOLDS = 2  # Use 2-fold CV for OOF generation

    # Reduce dimensionality to prevent errors with small sample sizes
    Config.MD_SVD_COMPONENTS = 5
    Config.CODE_SVD_COMPONENTS = 5

    # Reduce LightGBM complexity for speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    # Use a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the demo working directory
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_RIDGE_PATH = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")
    Config.MODEL_LGBM_PATH = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
    Config.VECTORIZER_MD_PATH = os.path.join(Config.WORKING_DIR, "tfidf_md.joblib")
    Config.VECTORIZER_CODE_PATH = os.path.join(Config.WORKING_DIR, "tfidf_code.joblib")
    Config.SVD_MD_PATH = os.path.join(Config.WORKING_DIR, "svd_md.joblib")
    Config.SVD_CODE_PATH = os.path.join(Config.WORKING_DIR, "svd_code.joblib")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    setup_reproducibility(Config.SEED)
    print("Configuration updated for fast execution.")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Data Loader ---")
    loader = NotebookDataLoader(debug=Config.DEBUG)

    # Load training data (force reload to skip cache)
    df_md, df_nb = loader.load_data(split="train", load_cached_data=False)

    print(f"Loaded {len(df_md)} markdown cells from {len(df_nb)} notebooks.")

    # Validations
    if df_md.empty or df_nb.empty:
        raise AssertionError("Data Loader returned empty DataFrames.")

    required_md_cols = ["id", "cell_id", "source", "rank"]
    if not all(col in df_md.columns for col in required_md_cols):
        raise AssertionError(
            f"Markdown DataFrame missing columns. Expected {required_md_cols}"
        )

    required_nb_cols = ["id", "code_context", "total_cells"]
    if not all(col in df_nb.columns for col in required_nb_cols):
        raise AssertionError(
            f"Notebook DataFrame missing columns. Expected {required_nb_cols}"
        )

    print("Data Loader validations passed.")

    # --------------------------------------------------------------------------
    # 3. Feature Extraction Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Feature Pipeline ---")
    fe_pipeline = FeaturePipeline()

    # Fit pipeline
    fe_pipeline.fit(df_md, df_nb, load_cached_data=False)

    # Transform Level 1 (Sparse TF-IDF)
    X_l1 = fe_pipeline.transform_level1(df_md)
    print(f"Level 1 Features Shape: {X_l1.shape}")

    if X_l1.shape[0] != len(df_md):
        raise AssertionError("Level 1 feature rows do not match input size.")

    # Transform Level 2 (Dense Stacked)
    # Mock some Level 1 predictions for the stacking feature
    mock_l1_preds = np.random.rand(len(df_md))
    df_l2 = fe_pipeline.transform_level2(df_md, df_nb, level1_preds=mock_l1_preds)
    print(f"Level 2 Features Shape: {df_l2.shape}")

    if len(df_l2) != len(df_md):
        raise AssertionError("Level 2 feature rows do not match input size.")

    expected_l2_cols = ["md_lsa_0", "code_lsa_0", "char_len", "md_ratio", "pred_ridge"]
    # Check if at least some expected columns are present
    if not any(c in df_l2.columns for c in expected_l2_cols):
        raise AssertionError("Level 2 DataFrame missing expected feature columns.")

    print("Feature Pipeline validations passed.")

    # --------------------------------------------------------------------------
    # 4. Model Training Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Model Classes ---")

    # Test Level 1 Ridge
    ridge = Level1Ridge()
    ridge.fit(X_l1, df_md["rank"])
    ridge_preds = ridge.predict(X_l1)

    if len(ridge_preds) != len(df_md):
        raise AssertionError("Ridge prediction length mismatch.")
    print("Level 1 Ridge training and prediction successful.")

    # Test Level 2 LightGBM
    lgbm = Level2GBM()
    # Create dummy validation set for API compatibility check
    lgbm.fit(X_train=df_l2, y_train=df_md["rank"], X_val=df_l2, y_val=df_md["rank"])
    lgbm_preds = lgbm.predict(df_l2)

    if len(lgbm_preds) != len(df_md):
        raise AssertionError("LightGBM prediction length mismatch.")
    print("Level 2 LightGBM training and prediction successful.")

    # --------------------------------------------------------------------------
    # 5. Pipeline Manager (End-to-End) Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Pipeline Manager (End-to-End) ---")
    pm = PipelineManager()

    # Run the full training routine
    # This will load data, generate OOF preds, train Ridge, create L2 features, train LGBM
    pm.train_stacking_ensemble(load_cached_data=False)

    # Check if models were saved
    if not os.path.exists(Config.MODEL_RIDGE_PATH):
        raise AssertionError("Ridge model file was not saved.")
    if not os.path.exists(Config.MODEL_LGBM_PATH):
        raise AssertionError("LightGBM model file was not saved.")

    # Run the inference routine
    # This will load test data, predict ranks, sort cells, and generate submission CSV
    pm.predict_and_sort(load_cached_data=False)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission CSV was not generated.")

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    if list(df_sub.columns) != ["id", "cell_order"]:
        raise AssertionError("Submission CSV has incorrect columns.")
    if df_sub.empty:
        raise AssertionError("Submission CSV is empty.")

    print(
        f"Pipeline execution successful. Submission generated at {Config.SUBMISSION_PATH}"
    )

    # --------------------------------------------------------------------------
    # 6. Metric Logic Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Metric Logic ---")
    # Synthetic Test Case
    # Ground Truth: A B C (Order: A=0, B=1, C=2)
    # Prediction: A C B (Order: A=0, C=1, B=2) -> Ranks mapped to GT: 0, 2, 1
    # Inversions in [0, 2, 1]: (2, 1) is one inversion.
    # Total pairs n(n-1) = 3*2 = 6.
    # Kendall Tau = 1 - 4 * (1 / 6) = 1 - 0.666... = 0.333...

    df_gt_dummy = pd.DataFrame(
        {"id": ["nb_test"], "cell_order": ["cell_A cell_B cell_C"]}
    )

    preds_dummy = {"nb_test": "cell_A cell_C cell_B"}

    score = kendall_tau_metric(df_gt_dummy, preds_dummy)
    expected_score = 1 - 4 * (1 / 6)

    print(f"Calculated Score: {score:.4f}, Expected: {expected_score:.4f}")

    if abs(score - expected_score) > 1e-6:
        raise AssertionError(
            f"Metric calculation incorrect. Got {score}, expected {expected_score}"
        )

    print("Metric logic verification passed.")

    print("\n=== Demonstration Complete ===")
