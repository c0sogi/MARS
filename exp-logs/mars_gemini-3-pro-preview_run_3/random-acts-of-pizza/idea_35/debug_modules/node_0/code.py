import os
import sys
import shutil
import numpy as np
import pandas as pd
from scipy import sparse

# Import from the provided library
from library.config import Config
from library.utils import set_seed, Timer
from library.data_manager import get_processed_data
from library.feature_engine import FeaturePipeline
from library.training_engine import TrainingEngine


def run_demo():
    print("Starting Pizza Request Prediction Pipeline Demo...")

    # =========================================================================
    # 1. OPTIMIZATION: Patch Config for Speed
    # =========================================================================
    print("Patching Configuration for fast demonstration...")

    # Reduce CV folds
    Config.N_FOLDS = 2

    # Reduce Vocabulary sizes to speed up Vectorization
    Config.VOCAB_UNIFIED = 100
    Config.VOCAB_LEXICAL = 100
    Config.VOCAB_COMMUNITY = 50

    # Reduce Model Complexity (Estimators/Iterations)
    # Random Forests
    Config.MODEL_UNIFIED_RF["n_estimators"] = 10
    Config.MODEL_LEXICAL_RF["n_estimators"] = 10
    Config.MODEL_COMMUNITY_RF["n_estimators"] = 10
    Config.MODEL_SEMANTIC_RF["n_estimators"] = 10

    # XGBoost
    Config.MODEL_SEMANTIC_XGB["n_estimators"] = 10
    Config.MODEL_SEMANTIC_XGB["early_stopping_rounds"] = 5

    # Logistic Regression
    Config.MODEL_METADATA_LR["max_iter"] = 50
    Config.MODEL_META_LR["max_iter"] = 50

    # Change Working Directory to avoid overwriting production cache if any
    Config.WORKING_DIR = "./working/demo_run_cache"
    Config.SUBMISSION_FILE = "./working/demo_submission/submission.csv"

    # Ensure clean state for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    set_seed(Config.RANDOM_STATE)

    # =========================================================================
    # 2. Data Loading & Preprocessing
    # =========================================================================
    print("\n--- Step 1: Data Loading & Preprocessing ---")

    # We force load_cached_data=False to demonstrate the processing logic
    train_df, val_df, test_df = get_processed_data(load_cached_data=False)

    # Verification
    assert not train_df.empty, "Train DataFrame is empty!"
    assert not val_df.empty, "Validation DataFrame is empty!"
    assert not test_df.empty, "Test DataFrame is empty!"
    assert "text_unified" in train_df.columns, "Feature 'text_unified' missing."

    print(f"Train Shape: {train_df.shape}")
    print(f"Val Shape:   {val_df.shape}")
    print(f"Test Shape:  {test_df.shape}")

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    print("\n--- Step 2: Feature Engineering ---")

    fe_pipeline = FeaturePipeline()

    # Generate all features (Sparse TF-IDF, Dense Embeddings, Metadata)
    # This uses the patched Config for vocab sizes
    feature_dict = fe_pipeline.process_all(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification of Feature Shapes
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    # Check Sparse Matrix (Unified View)
    assert feature_dict["X_train_unified"].shape[0] == n_train
    assert feature_dict["X_train_unified"].shape[1] <= Config.VOCAB_UNIFIED
    assert sparse.issparse(
        feature_dict["X_train_unified"]
    ), "Unified features should be sparse."

    # Check Dense Matrix (Semantic View)
    # Embedding dim for all-MiniLM-L6-v2 is 384
    assert feature_dict["X_train_semantic"].shape == (n_train, 384)
    assert not sparse.issparse(
        feature_dict["X_train_semantic"]
    ), "Semantic features should be dense."

    # Check Targets
    assert len(feature_dict["y_train"]) == n_train
    assert len(feature_dict["y_val"]) == n_val

    print("Feature generation successful. Shapes verified.")

    # =========================================================================
    # 4. Model Training (Level 1 CV + Level 2 Meta)
    # =========================================================================
    print("\n--- Step 3: Training Engine Execution ---")

    trainer = TrainingEngine()

    # This runs the full pipeline:
    # 1. CV on Train to get OOF preds
    # 2. Train Meta Learner on OOF
    # 3. Retrain Base Models on Train+Val
    # 4. Predict on Test
    trainer.run(feature_dict, load_cached_data=False)

    # =========================================================================
    # 5. Final Verification
    # =========================================================================
    print("\n--- Step 4: Final Verification ---")

    # Check Submission File
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_FILE}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check rows
    assert (
        len(sub_df) == n_test
    ), f"Submission row count mismatch. Expected {n_test}, got {len(sub_df)}"

    # Check columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"

    # Check probabilities
    probs = sub_df[Config.TARGET_COL]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range."
    assert probs.dtype == float, "Target column should be float."

    # Check OOF Cache existence
    oof_path = os.path.join(Config.WORKING_DIR, "level1_oof_predictions.npz")
    assert os.path.exists(oof_path), "OOF predictions cache not found."

    print("\nSUCCESS: Pipeline completed and verified.")
    print(f"Submission generated at: {Config.SUBMISSION_FILE}")
    print(f"Sample predictions:\n{sub_df.head()}")


if __name__ == "__main__":
    # Ensure execution within time limits
    with Timer("Full Demo Script"):
        run_demo()
