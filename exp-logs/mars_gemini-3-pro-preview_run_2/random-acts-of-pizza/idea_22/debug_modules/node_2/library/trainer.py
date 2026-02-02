import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from library.config import (
    SEED,
    N_FOLDS,
    N_JOBS,
    CACHE_DIR,
    SUBMISSION_PATH,
    LR_GRID,
    SVM_GRID,
    RIDGE_GRID,
    BAGGING_CONFIG,
)
from library.utils import set_seed, setup_logger
from library.data_loader import (
    load_labeled_data,
    load_test_data,
    extract_text_data,
    extract_numeric_data,
)
from library.feature_engineering import generate_embeddings, prepare_design_matrix
from library.model_factory import build_ensemble_pipeline

# Setup logger
logger = setup_logger(os.path.join(CACHE_DIR, "training.log"))


def tune_component(X, y, preprocessor, component_type, param_grid):
    """
    Helper function to tune a single component of the ensemble using GridSearchCV.
    Constructs a temporary pipeline matching the structure in model_factory.
    """
    if component_type == "lr":
        # Structure: Bagging -> LogisticRegression
        base = LogisticRegression(random_state=SEED)
        clf = BaggingClassifier(estimator=base, **BAGGING_CONFIG)
        # Param prefix: clf (Bagging) -> estimator (LR)
        prefix = "clf__estimator__"

    elif component_type == "svm":
        # Structure: Bagging -> Calibrated -> SGDClassifier
        base = SGDClassifier(random_state=SEED)
        calibrated = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
        clf = BaggingClassifier(estimator=calibrated, **BAGGING_CONFIG)
        # Param prefix: clf (Bagging) -> estimator (Calibrated) -> estimator (SGD)
        prefix = "clf__estimator__estimator__"

    elif component_type == "ridge":
        # Structure: Bagging -> Calibrated -> RidgeClassifier
        base = RidgeClassifier(random_state=SEED)
        calibrated = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
        clf = BaggingClassifier(estimator=calibrated, **BAGGING_CONFIG)
        # Param prefix: clf (Bagging) -> estimator (Calibrated) -> estimator (Ridge)
        prefix = "clf__estimator__estimator__"
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    # Build temporary pipeline for tuning
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    # Prefix the grid parameters to match the pipeline structure
    prefixed_grid = {f"{prefix}{k}": v for k, v in param_grid.items()}

    # Perform Grid Search
    # Using a smaller inner CV (3) for efficiency
    gs = GridSearchCV(
        pipeline, prefixed_grid, cv=3, scoring="roc_auc", n_jobs=N_JOBS, verbose=0
    )

    gs.fit(X, y)

    # Extract best parameters and remove prefix for clean return
    best_params = {}
    for k, v in gs.best_params_.items():
        param_name = k[len(prefix) :]
        best_params[param_name] = v

    return best_params, gs.best_score_


def generate_submission(metadata_start_idx, load_cached_data=True):
    """
    Generates predictions for the test set using the trained fold models.
    """
    logger.info("Loading test data for inference...")
    df_test = load_test_data(load_cached_data=load_cached_data)

    # Process Features
    text_data = extract_text_data(df_test)
    embeddings = generate_embeddings(
        text_data, "test", load_cached_data=load_cached_data
    )
    numeric_data = extract_numeric_data(df_test)
    X_test, _ = prepare_design_matrix(embeddings, numeric_data)

    logger.info(f"Test Design Matrix Shape: {X_test.shape}")

    # Aggregate Predictions
    test_preds_sum = np.zeros(len(df_test))
    models_found = 0

    for fold in range(N_FOLDS):
        model_path = os.path.join(CACHE_DIR, f"model_fold_{fold}.joblib")
        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold} not found. Skipping.")
            continue

        pipeline = joblib.load(model_path)
        fold_preds = pipeline.predict_proba(X_test)[:, 1]
        test_preds_sum += fold_preds
        models_found += 1

    if models_found == 0:
        raise RuntimeError("No trained models found for inference.")

    avg_preds = test_preds_sum / models_found

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
    )

    # Save
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main execution function for the training pipeline.
    """
    set_seed(SEED)
    logger.info("Starting training process...")

    # ---------------------------------------------------------
    # 1. Load and Prepare Data
    # ---------------------------------------------------------
    # Load Labeled Data
    df_train_full = load_labeled_data(load_cached_data=load_cached_data)
    y = df_train_full["requester_received_pizza"].values.astype(int)

    # Text Features
    logger.info("Processing text features...")
    text_data = extract_text_data(df_train_full)
    embeddings = generate_embeddings(
        text_data, "train", load_cached_data=load_cached_data
    )

    # Numeric Features
    logger.info("Processing numeric features...")
    numeric_data = extract_numeric_data(df_train_full)

    # Combine
    X, metadata_start_idx = prepare_design_matrix(embeddings, numeric_data)

    logger.info(f"Design Matrix Shape: {X.shape}")
    logger.info(f"Metadata starts at index: {metadata_start_idx}")

    # ---------------------------------------------------------
    # 2. Cross-Validation Loop
    # ---------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(y))
    fold_scores = []

    # Get a dummy pipeline just to extract the preprocessor logic
    # We clone it implicitly by rebuilding it or passing it to the pipeline constructor
    dummy_pipeline = build_ensemble_pipeline(metadata_start_idx)
    preprocessor_template = dummy_pipeline.named_steps["preprocessor"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # -----------------------------------------------------
        # Hyperparameter Tuning (Component-wise)
        # -----------------------------------------------------
        # Tune Logistic Regression
        logger.info("Tuning Logistic Regression component...")
        best_lr_params, lr_score = tune_component(
            X_train, y_train, preprocessor_template, "lr", LR_GRID
        )
        logger.info(f"Best LR Params: {best_lr_params} (CV Score: {lr_score})")

        # Tune SVM
        logger.info("Tuning SVM component...")
        best_svm_params, svm_score = tune_component(
            X_train, y_train, preprocessor_template, "svm", SVM_GRID
        )
        logger.info(f"Best SVM Params: {best_svm_params} (CV Score: {svm_score})")

        # Tune Ridge
        logger.info("Tuning Ridge component...")
        best_ridge_params, ridge_score = tune_component(
            X_train, y_train, preprocessor_template, "ridge", RIDGE_GRID
        )
        logger.info(f"Best Ridge Params: {best_ridge_params} (CV Score: {ridge_score})")

        # -----------------------------------------------------
        # Final Fold Training
        # -----------------------------------------------------
        logger.info("Training final ensemble for fold...")

        # Build pipeline with optimized parameters
        fold_pipeline = build_ensemble_pipeline(
            metadata_start_idx,
            lr_params=best_lr_params,
            svm_params=best_svm_params,
            ridge_params=best_ridge_params,
        )

        # Fit on full fold training data
        fold_pipeline.fit(X_train, y_train)

        # Save Model
        model_path = os.path.join(CACHE_DIR, f"model_fold_{fold}.joblib")
        joblib.dump(fold_pipeline, model_path)

        # Validate
        y_pred_proba = fold_pipeline.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = y_pred_proba

        score = roc_auc_score(y_val, y_pred_proba)
        fold_scores.append(score)
        logger.info(f"Fold {fold + 1} ROC AUC: {score}")

    # ---------------------------------------------------------
    # 3. Overall Evaluation
    # ---------------------------------------------------------
    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"Overall OOF ROC AUC: {overall_auc}")
    logger.info(f"Average Fold ROC AUC: {np.mean(fold_scores)}")

    # ---------------------------------------------------------
    # 4. Test Inference & Submission
    # ---------------------------------------------------------
    logger.info("Generating submission...")
    generate_submission(metadata_start_idx, load_cached_data)
