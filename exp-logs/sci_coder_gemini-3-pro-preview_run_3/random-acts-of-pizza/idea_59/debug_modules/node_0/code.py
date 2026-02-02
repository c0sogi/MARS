import os
import sys
import pandas as pd
import numpy as np

# 1. Import config and patch hyperparameters BEFORE importing other modules
# This ensures that modules importing these values get the updated versions.
import library.config as config

print("Configuring hyperparameters for fast demonstration...")
# Reduce Cross-Validation folds to minimum
config.N_FOLDS = 2

# Reduce Estimators for all models to ensure quick training
config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 5
config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 5
config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 5
config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 5
config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 5
config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 5

# Reduce Vectorizer complexity to speed up text processing
config.LEXICAL_VECTORIZER_PARAMS["max_features"] = 100
config.COMMUNITY_VECTORIZER_PARAMS["max_features"] = 50

# 2. Import remaining library modules
from library.utils import set_seed
from library.data_factory import load_union_dataset, load_test_dataset
from library.feature_engineering import FeaturePipeline
from library.pipeline_manager import PipelineManager


def main():
    print("Starting demonstration script...")

    # Set global seed for reproducibility
    set_seed(42)

    # --- Step 1: Data Loading ---
    print("\n[Step 1] Loading Data...")
    # We set load_cached_data=False to demonstrate the loading logic from metadata files
    df_train = load_union_dataset(load_cached_data=False)
    df_test = load_test_dataset(load_cached_data=False)

    # Validation
    assert not df_train.empty, "Training dataset is empty."
    assert not df_test.empty, "Test dataset is empty."
    assert config.TARGET_COL in df_train.columns, "Target column missing in train."
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    # --- Step 2: Feature Engineering ---
    print("\n[Step 2] Feature Engineering...")
    fp = FeaturePipeline()

    # Fit and Transform on Train
    # This processes metadata, text (TF-IDF), subreddits, and computes semantic embeddings
    print("Generating training features (this may take a moment for embeddings)...")
    X_train_dict = fp.fit_transform(df_train, load_cached_data=False)

    # Validate Feature Dictionary Structure
    expected_keys = ["metadata", "lexical", "behavioral", "semantic"]
    for key in expected_keys:
        assert key in X_train_dict, f"Missing feature group: {key}"
        assert X_train_dict[key].shape[0] == len(df_train), f"Shape mismatch for {key}"

    # Transform Test
    print("Generating test features...")
    X_test_dict = fp.transform(df_test, load_cached_data=False)

    for key in expected_keys:
        assert X_test_dict[key].shape[0] == len(df_test), f"Shape mismatch for {key}"

    # --- Step 3: Model Training & Pipeline ---
    print("\n[Step 3] Running Pipeline (CV & Training)...")
    pm = PipelineManager()
    y_train = df_train[config.TARGET_COL]

    # Run Cross-Validation to get Out-of-Fold (OOF) predictions
    # This uses the patched N_FOLDS=2
    oof_preds = pm.run_cv_and_oof(X_train_dict, y_train, load_cached_oof=False)

    # Validate OOF shape
    n_models = len(pm.all_model_classes)
    assert oof_preds.shape == (
        len(df_train),
        n_models,
    ), "OOF predictions shape mismatch"

    # Train Level 2 Meta-Learner on OOF predictions
    print("Training Meta-Learner...")
    pm.train_meta_learner(oof_preds, y_train)

    # Retrain 'Stable' Level 1 models on the full training dataset
    print("Retraining Stable Models on full dataset...")
    pm.retrain_stable_full(X_train_dict, y_train)

    # --- Step 4: Inference ---
    print("\n[Step 4] Generating Submission...")
    test_ids = df_test[config.ID_COL]

    # Generate predictions and save submission file
    pm.predict_and_submit(X_test_dict, test_ids)

    # Verify Submission File
    submission_path = config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission generated at: {submission_path}")
        print(sub_df.head())

        # Final Assertions
        assert len(sub_df) == len(df_test), "Submission row count mismatch"
        assert list(sub_df.columns) == [
            config.ID_COL,
            config.TARGET_COL,
        ], "Submission columns mismatch"
        assert (
            sub_df[config.TARGET_COL].dtype == float
        ), "Probability column should be float"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration finished successfully!")


if __name__ == "__main__":
    main()
