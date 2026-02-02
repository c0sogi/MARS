import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV

from library.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGETS_PATH,
    VAL_FEATURES_PATH,
    VAL_TARGETS_PATH,
    TEST_FEATURES_PATH,
    RIDGE_MODEL_PATH,
    SUBMISSION_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
    SEED,
    seed_everything,
)
from library.trainer import compute_spearman_metric


def train_ridge_head(load_cached_model=True):
    """
    Executes Stage 2: Linear Refinement using Ridge Regression.

    1. Loads cached interaction features (X) and targets (y).
    2. Trains a RidgeCV model (or loads cached model).
    3. Evaluates on Validation set.
    4. Generates predictions for Test set and saves submission.

    Args:
        load_cached_model (bool): If True, attempts to load a saved Ridge model
                                  instead of retraining.
    """
    seed_everything(SEED)
    print("Starting Stage 2: Ridge Regression Head Training...")

    # -------------------------------------------------------------------------
    # 1. Load Cached Features and Targets
    # -------------------------------------------------------------------------
    required_files = [
        TRAIN_FEATURES_PATH,
        TRAIN_TARGETS_PATH,
        VAL_FEATURES_PATH,
        VAL_TARGETS_PATH,
        TEST_FEATURES_PATH,
    ]

    for fpath in required_files:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Required feature file not found: {fpath}. "
                "Please run feature extraction (Stage 1 & Caching) first."
            )

    print("Loading cached features from disk...")
    X_train = np.load(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_TARGETS_PATH)
    X_val = np.load(VAL_FEATURES_PATH)
    y_val = np.load(VAL_TARGETS_PATH)
    X_test = np.load(TEST_FEATURES_PATH)

    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape:   {X_val.shape}")
    print(f"Test features shape:  {X_test.shape}")

    # -------------------------------------------------------------------------
    # 2. Model Training / Loading
    # -------------------------------------------------------------------------
    model = None

    if load_cached_model and os.path.exists(RIDGE_MODEL_PATH):
        print(f"Loading cached Ridge model from {RIDGE_MODEL_PATH}")
        model = joblib.load(RIDGE_MODEL_PATH)
    else:
        print("Training RidgeCV model...")
        # RidgeCV performs efficient Leave-One-Out Cross-Validation to select alpha
        # We use a range of alphas to handle different scales of regularization
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        model = RidgeCV(alphas=alphas)

        # Fit the model
        model.fit(X_train, y_train)

        # Save the model
        print(f"Saving trained Ridge model to {RIDGE_MODEL_PATH}")
        joblib.dump(model, RIDGE_MODEL_PATH)

        print(f"Best alpha(s): {model.alpha_}")

    # -------------------------------------------------------------------------
    # 3. Validation Evaluation
    # -------------------------------------------------------------------------
    print("Evaluating on Validation set...")
    val_preds = model.predict(X_val)

    # Clip predictions to valid probability range [0, 1]
    val_preds = np.clip(val_preds, 0, 1)

    # Compute metric
    val_score = compute_spearman_metric(val_preds, y_val)
    print(f"Stage 2 Validation Spearman Correlation: {val_score}")

    # -------------------------------------------------------------------------
    # 4. Test Inference & Submission
    # -------------------------------------------------------------------------
    print("Generating predictions for Test set...")
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 0, 1)

    # Load Test Metadata to get QA_IDs
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    # Ensure alignment
    if len(df_test_meta) != len(test_preds):
        raise ValueError(
            f"Mismatch between test metadata rows ({len(df_test_meta)}) "
            f"and predictions ({len(test_preds)})."
        )

    # Construct Submission DataFrame
    submission_df = pd.DataFrame(test_preds, columns=TARGET_COLS)
    submission_df.insert(0, "qa_id", df_test_meta["qa_id"])

    # Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print("Stage 2 completed successfully.")
