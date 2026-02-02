import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import from the provided library
from library.config import Config
from library.utils import load_data
from library.feature_extraction import FeatureManager
from library.ensemble_trainer import StackingEnsemble


def main():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration Override for Fast Demo
    # ==========================================
    # We modify the Config class attributes directly to optimize for speed
    # and demonstrate the pipeline on a small subset of data.

    print("Configuring hyperparameters for speed...")
    Config.set_seed(42)

    # Enable debug mode to use a small sample of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for quick demonstration

    # Reduce Cross-Validation folds
    Config.N_FOLDS = 2

    # Reduce Feature Dimensionality
    Config.TEXT_TFIDF_MAX_FEATURES = 50
    Config.SUBREDDIT_TFIDF_MAX_FEATURES = 20
    Config.SUBREDDIT_SVD_COMPONENTS = 5

    # Reduce Model Complexity (Level 1)
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead for small data

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = None  # Disable for this tiny run
    Config.XGB_PARAMS["n_jobs"] = 1

    # Reduce Meta-Learner Complexity (Level 2)
    Config.META_PARAMS["max_iter"] = 100

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nStep 1: Loading Data...")
    # We set load_cached_data=False to force the raw data processing pipeline
    train_df, val_df, test_df = load_data(load_cached_data=False, debug=Config.DEBUG)

    # Verification
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test DF size mismatch"
    assert Config.TARGET_COL in train_df.columns, "Target column missing in Train"
    print(f"Data loaded successfully. Train shape: {train_df.shape}")

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\nStep 2: Extracting Features...")
    # Instantiate the manager
    feature_manager = FeatureManager()

    # Extract features (this handles fitting on train and transforming all)
    # We disable cache loading here to demonstrate the extraction logic
    data_dict = feature_manager.extract_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification of Feature Shapes
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    # Check Lexical (Sparse)
    assert data_dict["X_train_lexical"].shape[0] == n_train
    assert data_dict["X_train_lexical"].shape[1] <= Config.TEXT_TFIDF_MAX_FEATURES

    # Check Behavioral (Sparse)
    assert data_dict["X_train_behavioral"].shape[0] == n_train

    # Check Dense (Semantic + SVD + Meta)
    # Expected dim: SBERT(384) + SVD(5) + Meta(~8)
    expected_dense_dim = 384 + Config.SUBREDDIT_SVD_COMPONENTS + 8
    # Note: Meta features count might vary slightly depending on implementation details
    # but should be consistent across splits.

    assert data_dict["X_train_dense"].shape[0] == n_train
    assert data_dict["X_test_dense"].shape[0] == n_test
    assert data_dict["X_train_dense"].shape[1] == data_dict["X_test_dense"].shape[1]

    # Check Targets
    assert data_dict["y_train"].shape[0] == n_train
    assert data_dict["y_val"].shape[0] == n_val

    print("Feature extraction complete and verified.")

    # ==========================================
    # 4. Model Training (Stacking Ensemble)
    # ==========================================
    print("\nStep 3: Training Stacking Ensemble...")
    ensemble = StackingEnsemble()

    # Fit the ensemble
    # This runs CV to train Level 1 models, generates OOF preds,
    # trains Level 2 meta-learner, and finally retrains Level 1 on full data.
    ensemble.fit(data_dict)

    print("Training complete.")

    # ==========================================
    # 5. Prediction and Submission
    # ==========================================
    print("\nStep 4: Generating Predictions...")
    predictions = ensemble.predict(data_dict)

    # Verification
    assert len(predictions) == n_test, "Prediction length mismatch"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Probabilities out of bounds"

    print("Saving Submission...")
    ensemble.save_submission(predictions)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (n_test, 2), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch"

    print(f"\nPipeline demonstration completed successfully.")
    print(f"Submission saved to: {submission_path}")
    print(f"First 5 predictions:\n{sub_df.head()}")


if __name__ == "__main__":
    main()
