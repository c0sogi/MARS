import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    N_FOLDS,
    SEED,
    TEMPORAL_COLS,
    USER_METADATA_COLS,
)
from library.utils import setup_logger, set_seed
from library.feature_engine import generate_features, ScalerWrapper
from library.model_factory import (
    create_bagged_logistic_ensemble,
    get_hyperparameter_grid,
)

# Initialize Logger
logger = setup_logger("runner", os.path.join(WORKING_DIR, "execution.log"))


def run_cv_training(debug=False, load_cached_features=True):
    """
    Orchestrates the 5-Fold Stratified Cross-Validation training pipeline.

    Steps:
    1. Load/Generate training features.
    2. Split data into 5 stratified folds.
    3. For each fold:
       a. Fit RankGauss Scaler on training data (Views 2, 3, 4).
       b. Transform training and validation data.
       c. Perform Grid Search for Bagged Logistic Regression.
       d. Train best model and evaluate on validation set.
       e. Save Scaler and Model.
    4. Aggregate and report metrics.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_features (bool): Whether to load features from cache.

    Returns:
        float: Average ROC AUC across all folds.
    """
    set_seed(SEED)
    logger.info(f"Starting CV Training (Debug={debug})...")

    # 1. Load Data
    df_train = generate_features(
        "train", load_cached_data=load_cached_features, debug=debug
    )

    # Identify feature columns (exclude metadata and target)
    target_col = "requester_received_pizza"
    meta_cols = ["request_id", target_col]
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    # Columns to scale (Views 3, 4) - View 2 (Structural) Removed
    cols_to_scale = TEMPORAL_COLS + USER_METADATA_COLS

    X = df_train
    y = df_train[target_col].values

    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processing Fold {fold + 1}/{N_FOLDS}...")

        # Split Data
        df_train_fold = X.iloc[train_idx].copy()
        df_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # 3a. Fit Scaler (RankGauss) on Training Fold
        # Only scale Views 2, 3, 4. View 1 (Embeddings) is already L2 normalized.
        scaler = ScalerWrapper(columns_to_scale=cols_to_scale)
        scaler.fit(df_train_fold)

        # 3b. Transform Data
        df_train_scaled = scaler.transform(df_train_fold)
        df_val_scaled = scaler.transform(df_val_fold)

        # Extract feature matrices for sklearn
        X_train_fold = df_train_scaled[feature_cols]
        X_val_fold = df_val_scaled[feature_cols]

        # 3c. Hyperparameter Tuning (Grid Search)
        # Create base model
        model = create_bagged_logistic_ensemble(random_state=SEED)
        param_grid = get_hyperparameter_grid()

        # Perform Grid Search
        # Using cv=3 internally for hyperparameter selection within the fold
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )

        logger.info(f"  Tuning hyperparameters for Fold {fold + 1}...")
        grid_search.fit(X_train_fold, y_train_fold)

        best_model = grid_search.best_estimator_
        logger.info(f"  Best Params: {grid_search.best_params_}")

        # 3d. Evaluation
        y_pred_proba = best_model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        fold_scores.append(score)

        logger.info(f"  Fold {fold + 1} ROC AUC: {score}")

        # 3e. Save Artifacts
        fold_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(fold_dir, exist_ok=True)

        joblib.dump(scaler, os.path.join(fold_dir, f"scaler_fold_{fold}.joblib"))
        joblib.dump(best_model, os.path.join(fold_dir, f"model_fold_{fold}.joblib"))

    mean_auc = np.mean(fold_scores)
    logger.info(f"CV Training Complete. Mean ROC AUC: {mean_auc}")

    return mean_auc


def generate_submission(debug=False, load_cached_features=True):
    """
    Generates predictions for the test set using the trained ensemble.

    Steps:
    1. Load/Generate test features.
    2. Load saved Scalers and Models for all 5 folds.
    3. For each fold:
       a. Apply fold-specific Scaler to test data.
       b. Predict probabilities using fold-specific Model.
    4. Average probabilities across folds.
    5. Save submission file.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_features (bool): Whether to load features from cache.
    """
    set_seed(SEED)
    logger.info(f"Starting Submission Generation (Debug={debug})...")

    # 1. Load Test Data
    df_test = generate_features(
        "test", load_cached_data=load_cached_features, debug=debug
    )

    # Identify feature columns (exclude metadata)
    # Note: Test set does not have target column, but might have request_id
    meta_cols = ["request_id"]
    feature_cols = [c for c in df_test.columns if c not in meta_cols]

    # Initialize array for aggregated predictions
    final_preds = np.zeros(len(df_test))

    # Directory where models are saved
    fold_dir = os.path.join(WORKING_DIR, "models")

    # 2. Iterate over folds
    for fold in range(N_FOLDS):
        logger.info(f"Predicting with Fold {fold + 1}/{N_FOLDS} model...")

        scaler_path = os.path.join(fold_dir, f"scaler_fold_{fold}.joblib")
        model_path = os.path.join(fold_dir, f"model_fold_{fold}.joblib")

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Artifacts for fold {fold} not found. Run training first."
            )

        # Load artifacts
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

        # 3a. Transform Test Data
        # IMPORTANT: We must apply the scaler fitted on this fold's training data
        # to ensure the distribution mapping (RankGauss) is consistent.
        df_test_scaled = scaler.transform(df_test)
        X_test_fold = df_test_scaled[feature_cols]

        # 3b. Predict
        preds = model.predict_proba(X_test_fold)[:, 1]
        final_preds += preds

    # 4. Average Predictions
    avg_preds = final_preds / N_FOLDS

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
    )

    # Save
    logger.info(f"Saving submission to {SUBMISSION_PATH}")
    submission.to_csv(SUBMISSION_PATH, index=False)

    logger.info("Submission generation complete.")
