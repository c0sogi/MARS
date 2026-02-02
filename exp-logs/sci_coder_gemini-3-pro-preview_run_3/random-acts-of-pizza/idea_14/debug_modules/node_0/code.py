import os
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.data_loader import load_and_preprocess_data
from library.feature_engine import FeaturePipeline
from library.model_stack import QuadStackingClassifier, run_stacking_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def optimize_config_for_demo():
    """
    Modifies the global Config parameters to ensure the demo runs quickly.
    """
    print("Optimization: Adjusting hyperparameters for fast execution...")

    # Reduce Random Forest estimators
    Config.RF_PARAMS["n_estimators"] = 5
    Config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead for small demo

    # Reduce XGBoost estimators and depth
    Config.XGB_PARAMS["n_estimators"] = 5
    Config.XGB_PARAMS["max_depth"] = 2
    Config.XGB_PARAMS["n_jobs"] = 1

    # Reduce Cross-Validation folds
    Config.N_FOLDS = 2

    # Reduce Vectorizer features to speed up TF-IDF
    Config.LEXICAL_VECTORIZER_PARAMS["max_features"] = 100
    Config.BEHAVIORAL_VECTORIZER_PARAMS["max_features"] = 50

    # Reduce SBERT batch size for safety
    Config.EMBEDDING_BATCH_SIZE = 16


def demo_data_loading():
    print("\n=== Demo: Data Loading ===")
    # Load data without cache to verify processing logic
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=False)

    # Verification
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    assert not train_df.empty, "Train dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"

    # Check if leakage columns are dropped
    leakage_cols = [
        c
        for c in train_df.columns
        if any(c.endswith(s) for s in Config.LEAKAGE_SUFFIXES)
    ]
    assert len(leakage_cols) == 0, f"Leakage columns found in train: {leakage_cols}"

    # Check if target exists in train/val but not test (test might have it if loaded from raw but here it's from metadata which usually has it, but let's check config)
    # The metadata/test.parquet usually doesn't have the target, or it's not used.
    # The data_loader ensures target is int if present.
    assert Config.TARGET_COL in train_df.columns, "Target column missing in train"

    return train_df, val_df, test_df


def demo_feature_engineering(train_df, val_df, test_df):
    print("\n=== Demo: Feature Engineering ===")

    # Initialize pipeline
    pipeline = FeaturePipeline()

    # Create views
    # We force load_cached_data=False to verify computation
    train_feats, val_feats, test_feats = pipeline.create_views(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification of dictionary structure
    expected_keys = {"lexical", "behavioral", "semantic", "contextual"}
    assert (
        set(train_feats.keys()) == expected_keys
    ), "Missing feature views in train features"

    # Verification of dimensions
    n_train = len(train_df)
    n_test = len(test_df)

    # Lexical (Sparse)
    assert train_feats["lexical"].shape[0] == n_train
    assert test_feats["lexical"].shape[0] == n_test

    # Semantic (Dense)
    assert train_feats["semantic"].shape[0] == n_train
    # SBERT (384) + Metadata (approx 13)
    assert (
        train_feats["semantic"].shape[1] >= 384
    ), "Semantic features dimensionality unexpected"

    print("Feature views generated and verified successfully.")
    return train_feats, val_feats, test_feats


def demo_model_stacking(train_feats, train_y, test_feats):
    print("\n=== Demo: Model Stacking ===")

    # Initialize Stacker
    stacker = QuadStackingClassifier()

    # Fit
    print("Fitting stacker (this involves CV and base learner training)...")
    stacker.fit(train_feats, train_y)
    assert stacker.models_fitted is True, "Model fitted flag not set"

    # Predict
    print("Predicting on test set...")
    probs = stacker.predict_proba(test_feats)

    # Verification
    assert len(probs) == test_feats["contextual"].shape[0], "Prediction length mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print(f"Predictions generated. Mean probability: {np.mean(probs):.4f}")
    return probs


def demo_full_pipeline_wrapper(train_feats, train_y, test_feats, test_ids):
    print("\n=== Demo: Full Pipeline Wrapper ===")

    # Run the provided wrapper function
    submission_df = run_stacking_pipeline(train_feats, train_y, test_feats, test_ids)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    assert len(submission_df) == len(test_ids), "Submission row count mismatch"
    assert Config.ID_COL in submission_df.columns, "ID column missing in submission"
    assert (
        Config.TARGET_COL in submission_df.columns
    ), "Target column missing in submission"

    print("Full pipeline wrapper executed successfully.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    optimize_config_for_demo()

    # 2. Data Loading
    train_df, val_df, test_df = demo_data_loading()

    # Prepare targets and IDs
    y_train = train_df[Config.TARGET_COL].values
    test_ids = test_df[Config.ID_COL].values

    # 3. Feature Engineering
    # Note: We pass val_df to create_views as required by the signature,
    # but for the final training in this demo, we might just use train_feats.
    # In a real scenario, one might concatenate train and val for final training,
    # or keep them separate. The provided stacker splits the training data internally
    # using K-Fold, so we pass the main training set.
    train_feats, val_feats, test_feats = demo_feature_engineering(
        train_df, val_df, test_df
    )

    # 4. Model Stacking (Direct Usage)
    demo_model_stacking(train_feats, y_train, test_feats)

    # 5. Full Pipeline (Wrapper Usage)
    demo_full_pipeline_wrapper(train_feats, y_train, test_feats, test_ids)

    print("\nAll demonstrations completed successfully.")
