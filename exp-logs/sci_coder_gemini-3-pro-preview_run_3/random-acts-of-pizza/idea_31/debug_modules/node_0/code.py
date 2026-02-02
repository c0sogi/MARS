import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_and_process_data
from library.models import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
    StackingMetaLearner,
)
from library.train import run_training


def main():
    print("Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Patch Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce model complexity for the demo
    Config.SPARSE_RF_PARAMS["n_estimators"] = 5
    Config.DENSE_RF_PARAMS["n_estimators"] = 5
    Config.DENSE_RF_PARAMS["max_depth"] = 5
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_EARLY_STOPPING_ROUNDS = 5
    Config.LINEAR_PARAMS["max_iter"] = 50

    # Set seed for reproducibility
    set_seed(Config.RANDOM_SEED)
    print("Configuration patched. Random seed set.")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading and Feature Engineering...")

    # Load data in debug mode (slices to small subset)
    debug_size = 50
    data = load_and_process_data(
        load_cached_data=False, debug=True, debug_size=debug_size
    )

    # Verify dictionary structure
    assert "train" in data, "Data dict missing 'train' key"
    assert "val" in data, "Data dict missing 'val' key"
    assert "test" in data, "Data dict missing 'test' key"

    # Verify Train Data Shapes
    train_data = data["train"]
    n_samples = train_data["y"].shape[0]
    print(f"Loaded {n_samples} training samples (Debug mode).")

    assert n_samples <= debug_size, f"Debug slicing failed, got {n_samples} samples"
    assert sp.issparse(train_data["lexical"]), "Lexical features should be sparse"
    assert sp.issparse(train_data["behavioral"]), "Behavioral features should be sparse"
    assert isinstance(
        train_data["semantic"], np.ndarray
    ), "Semantic features should be dense numpy array"
    assert isinstance(
        train_data["metadata"], np.ndarray
    ), "Metadata features should be dense numpy array"

    # Check dimensions match
    assert train_data["lexical"].shape[0] == n_samples
    assert train_data["metadata"].shape[0] == n_samples

    print("Data structure and types verified.")

    # -------------------------------------------------------------------------
    # 3. Model Component Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Individual Model Components...")

    # Extract features for model testing
    X_lex = train_data["lexical"]
    X_beh = train_data["behavioral"]
    X_sem = train_data["semantic"]
    X_meta = train_data["metadata"]
    y = train_data["y"]

    # 3.1 LexicalBagger (Sparse RF)
    print("Testing LexicalBagger...")
    model_lex = LexicalBagger()
    model_lex.fit(X_lex, X_meta, y)
    preds_lex = model_lex.predict_proba(X_lex, X_meta)
    assert preds_lex.shape == (n_samples,), "LexicalBagger prediction shape mismatch"
    assert np.all(
        (preds_lex >= 0) & (preds_lex <= 1)
    ), "Predictions out of probability range"

    # 3.2 CommunityBagger (Sparse RF)
    print("Testing CommunityBagger...")
    model_comm = CommunityBagger()
    model_comm.fit(X_beh, X_meta, y)
    preds_comm = model_comm.predict_proba(X_beh, X_meta)
    assert preds_comm.shape == (n_samples,), "CommunityBagger prediction shape mismatch"

    # 3.3 SemanticBooster (XGBoost)
    print("Testing SemanticBooster...")
    model_xgb = SemanticBooster()
    # Create a dummy validation set for early stopping test
    eval_set = (X_sem, X_meta, y)
    model_xgb.fit(X_sem, X_meta, y, eval_set=eval_set)
    preds_xgb = model_xgb.predict_proba(X_sem, X_meta)
    assert preds_xgb.shape == (n_samples,), "SemanticBooster prediction shape mismatch"

    # 3.4 SemanticBagger (Dense RF)
    print("Testing SemanticBagger...")
    model_sem_rf = SemanticBagger()
    model_sem_rf.fit(X_sem, X_meta, y)
    preds_sem_rf = model_sem_rf.predict_proba(X_sem, X_meta)
    assert preds_sem_rf.shape == (
        n_samples,
    ), "SemanticBagger prediction shape mismatch"

    # 3.5 MetadataAnchor (Logistic Regression)
    print("Testing MetadataAnchor...")
    model_meta = MetadataAnchor()
    model_meta.fit(X_meta, y)
    preds_meta = model_meta.predict_proba(X_meta)
    assert preds_meta.shape == (n_samples,), "MetadataAnchor prediction shape mismatch"

    # 3.6 StackingMetaLearner
    print("Testing StackingMetaLearner...")
    # Create dummy level 1 predictions
    X_level1 = np.column_stack(
        [preds_lex, preds_comm, preds_xgb, preds_sem_rf, preds_meta]
    )
    meta_learner = StackingMetaLearner()
    meta_learner.fit(X_level1, y)
    final_preds = meta_learner.predict_proba(X_level1)
    assert final_preds.shape == (n_samples,), "MetaLearner prediction shape mismatch"

    print("All model components verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Full Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[4] Running Full Training Pipeline (End-to-End)...")

    # Run the pipeline function provided in library/train.py
    # This handles CV, OOF generation, Meta-training, Retraining, and Submission
    run_training(debug=True, debug_size=debug_size)

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")

    # Verify submission content
    assert Config.ID_COL in df_sub.columns
    assert Config.TARGET_COL in df_sub.columns
    assert df_sub[Config.TARGET_COL].min() >= 0.0
    assert df_sub[Config.TARGET_COL].max() <= 1.0

    # In debug mode, we sliced the test set too, so rows should match debug_size (or less if test set is smaller)
    # The test set has 1162 rows, debug_size is 50.
    assert len(df_sub) <= debug_size

    print("\nFull pipeline executed successfully.")
    print("Demonstration Complete.")


if __name__ == "__main__":
    main()
