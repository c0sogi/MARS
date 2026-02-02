import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from library.config import PathConfig, TrainConfig, TARGET_COLS
from library.dataset import load_data
from library.utils import (
    seed_everything,
    compute_spearmanr,
    save_joblib,
    load_joblib,
)


def train_ridge_and_predict(load_cached_model=True):
    """
    Trains a Ridge Regression model on extracted features (or loads a cached one),
    evaluates it on the validation set, and generates the submission file.

    Args:
        load_cached_model (bool): If True, attempts to load a trained model from disk
                                  instead of retraining.

    Returns:
        None
    """
    seed_everything(TrainConfig.seed)

    # -------------------------------------------------------------------------
    # 1. Load Features and Targets
    # -------------------------------------------------------------------------
    # We assume these files exist because feature_pipeline.py must be run first.
    if not (
        os.path.exists(PathConfig.TRAIN_FEATURES_CACHE)
        and os.path.exists(PathConfig.TRAIN_TARGETS_CACHE)
        and os.path.exists(PathConfig.VAL_FEATURES_CACHE)
        and os.path.exists(PathConfig.VAL_TARGETS_CACHE)
        and os.path.exists(PathConfig.TEST_FEATURES_CACHE)
    ):
        raise FileNotFoundError(
            "Feature cache files not found. Please run the feature extraction pipeline first."
        )

    print("Loading features from cache...")
    train_features = np.load(PathConfig.TRAIN_FEATURES_CACHE)
    train_targets = np.load(PathConfig.TRAIN_TARGETS_CACHE)
    val_features = np.load(PathConfig.VAL_FEATURES_CACHE)
    val_targets = np.load(PathConfig.VAL_TARGETS_CACHE)
    test_features = np.load(PathConfig.TEST_FEATURES_CACHE)

    # -------------------------------------------------------------------------
    # 2. Train or Load Model
    # -------------------------------------------------------------------------
    model = None
    model_path = PathConfig.RIDGE_SAVE_PATH

    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached Ridge model from {model_path}...")
        try:
            model = load_joblib(model_path)
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")
            model = None

    if model is None:
        print(f"Training RidgeCV model with alphas={TrainConfig.ridge_alphas}...")
        # RidgeCV efficiently performs Leave-One-Out Cross-Validation to select alpha
        model = RidgeCV(
            alphas=list(TrainConfig.ridge_alphas),
            fit_intercept=True,
            scoring=None,  # Default is R^2, which is fine for minimizing MSE
        )
        model.fit(train_features, train_targets)

        print(f"Saving Ridge model to {model_path}...")
        save_joblib(model, model_path)

    # -------------------------------------------------------------------------
    # 3. Validation
    # -------------------------------------------------------------------------
    print("Evaluating on validation set...")
    val_preds = model.predict(val_features)

    # Clip predictions to valid probability range [0, 1]
    val_preds = np.clip(val_preds, 0.0, 1.0)

    score = compute_spearmanr(val_targets, val_preds)
    print(f"Validation Spearman Correlation: {score}")

    # -------------------------------------------------------------------------
    # 4. Test Prediction and Submission
    # -------------------------------------------------------------------------
    print("Generating test predictions...")
    test_preds = model.predict(test_features)
    test_preds = np.clip(test_preds, 0.0, 1.0)

    # Load Test Metadata to get qa_id
    # The order of test_features corresponds to rows in test_metadata.csv
    # We use load_data to ensure consistency if we are working with cached subsets
    _, _, test_df = load_data(load_cached_data=True)

    if len(test_df) != len(test_preds):
        raise ValueError(
            f"Mismatch between test metadata rows ({len(test_df)}) "
            f"and predictions ({len(test_preds)})."
        )

    # Construct Submission DataFrame
    submission_df = pd.DataFrame(test_preds, columns=TARGET_COLS)
    submission_df.insert(0, "qa_id", test_df["qa_id"])

    # Save Submission
    print(f"Saving submission to {PathConfig.SUBMISSION_FILE}...")
    submission_df.to_csv(PathConfig.SUBMISSION_FILE, index=False)
    print("Submission generation complete.")
