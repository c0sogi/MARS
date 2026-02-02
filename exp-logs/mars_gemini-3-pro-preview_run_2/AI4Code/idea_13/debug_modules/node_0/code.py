import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library components
from library.config import Config
from library.data_loader import NotebookProcessor, load_metadata
from library.stage1_ridge import Stage1Ridge
from library.stage2_metric import Stage2Metric
from library.stage3_lgbm import Stage3LGBM
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Pipeline Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    # We subclass Config to create a lightweight configuration for demonstration.
    # This ensures the code runs quickly on a small subset of data.
    class DemoConfig(Config):
        # Paths
        WORKING_DIR = "./working/demo_run"
        SUBMISSION_DIR = "./working/demo_submission"
        SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

        # Ensure clean state
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Update artifact paths to point to the demo directory
        TFIDF_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
        SVD_PATH = os.path.join(WORKING_DIR, "svd_model.joblib")
        RIDGE_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")
        METRIC_MODEL_PATH = os.path.join(WORKING_DIR, "metric_model.pth")

        TRAIN_DATAFRAME_PATH = os.path.join(WORKING_DIR, "train_dataframe.parquet")
        VAL_DATAFRAME_PATH = os.path.join(WORKING_DIR, "val_dataframe.parquet")
        TEST_DATAFRAME_PATH = os.path.join(WORKING_DIR, "test_dataframe.parquet")

        TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
        VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
        TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

        TRAIN_RIDGE_OOF_PATH = os.path.join(WORKING_DIR, "stage1_oof_preds.parquet")
        VAL_RIDGE_PREDS_PATH = os.path.join(WORKING_DIR, "stage1_val_preds.parquet")
        TEST_RIDGE_PREDS_PATH = os.path.join(WORKING_DIR, "stage1_test_preds.parquet")

        # Speed Optimizations
        DEBUG = True
        DEBUG_SAMPLE_SIZE = 200  # Process only 200 notebooks

        # Text Vectorization
        VOCAB_SIZE = 1000
        MIN_DF = 1  # Allow rare tokens in small sample
        SVD_COMPONENTS = 16  # Reduced dimension
        SVD_ITER = 2

        # Stage 2 (Metric Learning)
        METRIC_INPUT_DIM = 16  # Must match SVD_COMPONENTS
        METRIC_HIDDEN_DIM = 32
        METRIC_EMBEDDING_DIM = 16
        METRIC_EPOCHS = 1
        METRIC_BATCH_SIZE = 32

        # Stage 3 (LightGBM)
        LGBM_PARAMS = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 7,
            "n_estimators": 10,  # Very few trees for demo
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 1,
        }
        LGBM_EARLY_STOPPING_ROUNDS = 5

    # Set global seed
    seed_everything(DemoConfig.SEED)
    print(f"Configuration initialized. Working dir: {DemoConfig.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Data Loading ---")
    processor = NotebookProcessor(DemoConfig)

    # Load Train (Debug subset)
    df_train = processor.load_data("train", load_cached_data=False)
    print(
        f"Loaded {len(df_train)} training cells from {DemoConfig.DEBUG_SAMPLE_SIZE} notebooks."
    )

    # Validation
    required_cols = ["id", "cell_id", "cell_type", "source", "norm_rank"]
    for col in required_cols:
        if col not in df_train.columns:
            raise AssertionError(f"Missing column '{col}' in processed dataframe")

    # Load Val and Test to ensure pipeline readiness
    df_val = processor.load_data("val", load_cached_data=False)
    df_test = processor.load_data("test", load_cached_data=False)
    print("Data loading verified.")

    # --------------------------------------------------------------------------
    # 3. Stage 1: Ridge Regression
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Stage 1 (Ridge Regression) ---")
    stage1 = Stage1Ridge(DemoConfig)

    # Run Stage 1 (Train OOF + Inference on Val/Test)
    df_train_s1, df_val_s1, df_test_s1 = stage1.run(load_cached_preds=False)

    # Validation
    if "ridge_pred" not in df_train_s1.columns:
        raise AssertionError("Ridge predictions missing from training dataframe")

    if not os.path.exists(DemoConfig.RIDGE_PATH):
        raise AssertionError("Ridge model file was not saved")

    print("Stage 1 completed and verified.")

    # --------------------------------------------------------------------------
    # 4. Stage 2: Metric Learning (Siamese Network)
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Stage 2 (Metric Learning) ---")
    stage2 = Stage2Metric(DemoConfig)

    # Train the model
    stage2.train()

    # Validation: Check model artifact
    if not os.path.exists(DemoConfig.METRIC_MODEL_PATH):
        raise AssertionError("Metric learning model file was not saved")

    # Validation: Check inference
    test_texts = ["import numpy as np", "# This is a markdown cell"]
    embeddings = stage2.get_projected_embeddings(test_texts)

    if embeddings.shape != (2, DemoConfig.METRIC_EMBEDDING_DIM):
        raise AssertionError(
            f"Embedding shape mismatch. Expected (2, {DemoConfig.METRIC_EMBEDDING_DIM}), got {embeddings.shape}"
        )

    print("Stage 2 completed and verified.")

    # --------------------------------------------------------------------------
    # 5. Stage 3: LightGBM Ranking
    # --------------------------------------------------------------------------
    print("\n--- Step 5: Stage 3 (LightGBM Ranking) ---")
    stage3 = Stage3LGBM(DemoConfig)

    # Train LightGBM
    # Note: This internally calls stage2.get_projected_embeddings and generates anchor features
    stage3.train(load_cached_features=False)

    # Validation: Check model artifact
    lgbm_model_path = os.path.join(DemoConfig.WORKING_DIR, "lgbm_model.txt")
    if not os.path.exists(lgbm_model_path):
        raise AssertionError("LightGBM model file was not saved")

    # Generate Submission
    stage3.predict(load_cached_features=False)

    # Validation: Check submission file
    if not os.path.exists(DemoConfig.SUBMISSION_PATH):
        raise AssertionError("Submission file was not generated")

    df_sub = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    if list(df_sub.columns) != ["id", "cell_order"]:
        raise AssertionError(f"Invalid submission columns: {df_sub.columns}")

    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty")

    print(f"Stage 3 completed. Submission generated at {DemoConfig.SUBMISSION_PATH}")
    print(f"Submission head:\n{df_sub.head(3)}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
