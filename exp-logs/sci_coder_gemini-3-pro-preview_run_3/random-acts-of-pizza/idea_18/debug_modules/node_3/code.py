import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------------------------
# 1. Configuration Override for Speed
# ------------------------------------------------------------------------------
# We import Config first to patch it before other modules use it.
from library.config import Config

print("Configuring pipeline for fast demonstration...")

# Redirect working directories to a demo folder to avoid conflicts
Config.WORKING_DIR = "./working/demo_pipeline"
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Reduce computational load for the demo
Config.N_FOLDS = 2  # Reduce CV folds
Config.EARLY_STOPPING_ROUNDS = 2

# Reduce Base Learner complexity
Config.RF_LEXICAL_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
Config.RF_SEMANTIC_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
Config.RF_BEHAVIORAL_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
Config.XGB_SEMANTIC_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
Config.XGB_BEHAVIORAL_PARAMS.update({"n_estimators": 10, "n_jobs": 1})

# Reduce Feature dimensionality
Config.TEXT_TFIDF_PARAMS["max_features"] = 50
Config.SUBREDDIT_TFIDF_PARAMS["max_features"] = 20

# ------------------------------------------------------------------------------
# 2. Import Library Modules
# ------------------------------------------------------------------------------
from library.data_utils import load_datasets
from library.features import MultiModalFeatureGenerator
from library.training import StackingEngine

# ------------------------------------------------------------------------------
# 3. Execution Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== 1. Loading Data ===")
    # Load a small sample (50 rows) to ensure quick execution
    # load_cached_data=False ensures we run the preprocessing logic
    train_df, val_df, test_df = load_datasets(load_cached_data=False, sample_size=50)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Validate Data Loading
    assert len(train_df) == 50, "Training set sampling failed."
    assert Config.SUBREDDIT_COL in train_df.columns, "Subreddit column missing."
    assert isinstance(
        train_df[Config.SUBREDDIT_COL].iloc[0], str
    ), "Subreddit serialization failed."

    print("\n=== 2. Feature Generation ===")
    feature_gen = MultiModalFeatureGenerator()

    # Fit on training data
    print("Fitting feature generator...")
    feature_gen.fit(train_df)

    # Transform all splits
    # We disable loading from cache to demonstrate the generation process
    print("Transforming Train sets...")
    X_train = feature_gen.transform(train_df, "train", load_cached_data=False)

    print("Transforming Val sets...")
    X_val = feature_gen.transform(val_df, "val", load_cached_data=False)

    print("Transforming Test sets...")
    X_test = feature_gen.transform(test_df, "test", load_cached_data=False)

    # Validate Feature Shapes
    # Check if keys exist
    expected_keys = ["lexical", "semantic", "community", "persona", "meta"]
    for key in expected_keys:
        assert key in X_train, f"Missing feature key: {key}"
        assert X_train[key].shape[0] == 50, f"Feature row count mismatch for {key}"

    print("Feature generation successful.")

    print("\n=== 3. Model Training (Stacking) ===")
    engine = StackingEngine()

    # Extract targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # A. Level 1 Cross-Validation (OOF Generation)
    print("Running Level 1 Cross-Validation...")
    oof_preds = engine.train_level_1_cv(X_train, y_train)

    assert oof_preds.shape == (
        50,
        6,
    ), f"OOF shape mismatch. Expected (50, 6), got {oof_preds.shape}"

    # B. Level 2 Meta-Learner Training
    print("Training Level 2 Meta-Learner...")
    engine.train_level_2(oof_preds, y_train)

    # C. Final Retraining of Base Learners
    print("Retraining Base Learners on full data...")
    engine.retrain_level_1_final(X_train, y_train, X_val, y_val)

    # Validate Model Artifacts
    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "base_models.joblib")
    ), "Base models not saved."
    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "meta_model.joblib")
    ), "Meta model not saved."

    print("\n=== 4. Prediction & Submission ===")
    # Generate predictions on test set
    test_preds = engine.predict(X_test)

    assert len(test_preds) == 50, "Prediction count mismatch."
    assert np.all((test_preds >= 0) & (test_preds <= 1)), "Probabilities out of bounds."

    # Create submission dataframe
    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": test_preds}
    )

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final Verification
    saved_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("\nSubmission Head:")
    print(saved_sub.head())

    assert saved_sub.shape == (50, 2), "Submission file shape incorrect."
    assert list(saved_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect."

    print("\n=== Demo Completed Successfully ===")
