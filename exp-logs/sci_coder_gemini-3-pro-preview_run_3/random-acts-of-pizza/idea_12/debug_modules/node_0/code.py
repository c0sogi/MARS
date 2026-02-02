import os
import sys
import numpy as np
import pandas as pd
import unittest.mock
from sklearn.metrics import roc_auc_score

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as fe
import library.model_definitions as models
import library.trainer as trainer


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override hyperparameters in the config module to run quickly
    # These changes affect the models when they are instantiated later
    config.RF_ESTIMATORS = 5
    config.XGB_ESTIMATORS = 5
    config.XGB_EARLY_STOPPING_ROUNDS = 2
    config.N_FOLDS = 2

    # Reduce dimensionality for feature engineering speed
    config.TEXT_TFIDF_MAX_FEATURES = 50
    config.SUBREDDIT_TFIDF_MAX_FEATURES = 50

    # Set seed for reproducibility
    utils.set_seed(42)
    print("Configuration updated: RF_ESTIMATORS=5, N_FOLDS=2, etc.")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subsetting
    # -------------------------------------------------------------------------
    print("\n[2] Loading and subsetting data...")

    # Load actual data (ignoring cache to ensure we see the loading process)
    # We use the library function to load the dataframes
    full_train_df, full_val_df, full_test_df = data_loader.load_dataset(
        load_cached_data=False
    )

    # Create small subsets for the demonstration
    SUBSET_SIZE = 40
    train_subset = full_train_df.iloc[:SUBSET_SIZE].copy()
    val_subset = full_val_df.iloc[:SUBSET_SIZE].copy()
    test_subset = full_test_df.iloc[:SUBSET_SIZE].copy()

    # Ensure target diversity in subsets to prevent model training errors with XGBoost/RF
    target_col = config.TARGET_COL
    if train_subset[target_col].nunique() < 2:
        train_subset.iloc[0, train_subset.columns.get_loc(target_col)] = 0
        train_subset.iloc[1, train_subset.columns.get_loc(target_col)] = 1

    if val_subset[target_col].nunique() < 2:
        val_subset.iloc[0, val_subset.columns.get_loc(target_col)] = 0
        val_subset.iloc[1, val_subset.columns.get_loc(target_col)] = 1

    print(
        f"Subset Shapes -> Train: {train_subset.shape}, Val: {val_subset.shape}, Test: {test_subset.shape}"
    )

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Running Feature Engineering Pipeline...")

    # Instantiate the pipeline
    pipeline = fe.FeaturePipeline()

    # Fit on the training subset
    print("Fitting pipeline on train subset...")
    pipeline.fit(train_subset)

    # Transform datasets
    print("Transforming subsets...")
    train_views = pipeline.transform(train_subset)
    val_views = pipeline.transform(val_subset)
    test_views = pipeline.transform(test_subset)

    # Validate outputs
    for view_name in ["lexical", "behavioral", "semantic"]:
        assert view_name in train_views, f"Missing view: {view_name}"
        assert (
            train_views[view_name].shape[0] == SUBSET_SIZE
        ), f"Shape mismatch in {view_name}"

    print("Feature Engineering successful. Shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Training (Component Level)
    # -------------------------------------------------------------------------
    print("\n[4] Testing Individual Model Components...")

    y_train = train_subset[target_col].values
    y_val = val_subset[target_col].values

    # --- Lexical RF ---
    print("Testing LexicalRF...")
    lex_rf = models.LexicalRF()
    lex_rf.fit(train_views["lexical"], y_train)
    p_lex = lex_rf.predict_proba(val_views["lexical"])
    assert p_lex.shape == (SUBSET_SIZE, 2), "LexicalRF prediction shape mismatch"

    # --- Behavioral RF ---
    print("Testing BehavioralRF...")
    beh_rf = models.BehavioralRF()
    beh_rf.fit(train_views["behavioral"], y_train)
    p_beh = beh_rf.predict_proba(val_views["behavioral"])
    assert p_beh.shape == (SUBSET_SIZE, 2), "BehavioralRF prediction shape mismatch"

    # --- Semantic XGB ---
    print("Testing SemanticXGB...")
    sem_xgb = models.SemanticXGB()
    sem_xgb.fit(
        train_views["semantic"], y_train, X_val=val_views["semantic"], y_val=y_val
    )
    p_sem = sem_xgb.predict_proba(val_views["semantic"])
    assert p_sem.shape == (SUBSET_SIZE, 2), "SemanticXGB prediction shape mismatch"

    # --- Meta Learner ---
    print("Testing MetaLearner...")
    # Create dummy Level 1 outputs
    X_meta = np.column_stack([p_lex[:, 1], p_beh[:, 1], p_sem[:, 1]])
    meta_learner = models.MetaLearner()
    meta_learner.fit(X_meta, y_val)
    p_meta = meta_learner.predict_proba(X_meta)
    assert p_meta.shape == (SUBSET_SIZE, 2), "MetaLearner prediction shape mismatch"

    print("All model components verified.")

    # -------------------------------------------------------------------------
    # 5. Full Trainer Execution (Integration Test)
    # -------------------------------------------------------------------------
    print("\n[5] Running Full Trainer Pipeline (Mocked Data)...")

    # We mock `load_dataset` in `library.trainer` to return our subsets.
    # This allows us to run the full `train_stacking_ensemble` logic without
    # processing the entire dataset, which would be slow.

    with unittest.mock.patch("library.trainer.load_dataset") as mock_load:
        # Configure the mock to return our subsets
        mock_load.return_value = (train_subset, val_subset, test_subset)

        # Run the trainer
        # Note: load_cached_data=False ensures we don't accidentally pick up old large files
        # though our mock intercepts the call anyway.
        trainer.train_stacking_ensemble(load_cached_data=False)

    # -------------------------------------------------------------------------
    # 6. Verification of Results
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission...")

    submission_path = config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    submission_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {submission_df.shape}")
    print(submission_df.head())

    # Verify content
    assert (
        len(submission_df) == SUBSET_SIZE
    ), f"Expected {SUBSET_SIZE} predictions, got {len(submission_df)}"
    assert config.ID_COL in submission_df.columns, "Missing ID column"
    assert config.TARGET_COL in submission_df.columns, "Missing Target column"

    # Verify probabilities
    probs = submission_df[config.TARGET_COL]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
