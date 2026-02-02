import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import setup_logger, save_object, load_object
from library.feature_extraction import FeaturePreprocessor
from library.model_factory import create_pipeline

# Initialize logger
logger = setup_logger("trainer")


def run_training(debug=False):
    """
    Executes the 5-Fold Stratified Cross-Validation training loop with internal Grid Search.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.

    Returns:
        list: Paths to the saved model files for each fold.
    """
    logger.info(f"Starting training run (Debug={debug})...")

    # 1. Load Training Data
    preprocessor = FeaturePreprocessor()
    data = preprocessor.get_data(split="train", load_cached=True, debug=debug)

    X = data["X"]
    y = data["y"]
    feature_slices = data["feature_slices"]

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Placeholder for Out-of-Fold Predictions
    oof_preds = np.zeros(len(y))
    model_paths = []

    # Directory for saving models
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold + 1} / {Config.N_FOLDS} {'='*20}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create Base Pipeline
        # We initialize with default parameters; GridSearchCV will override them.
        pipeline = create_pipeline(
            feature_slices=feature_slices,
            pca_components=Config.AUX_PCA_COMPONENTS,
            n_bagging_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
        )

        # Define Hyperparameter Grid for the Base Estimator (Logistic Regression)
        # Note: The pipeline structure is 'preprocessor' -> 'classifier' (BaggingClassifier).
        # BaggingClassifier holds the base estimator in the 'estimator' attribute.
        param_grid = {
            "classifier__estimator__C": Config.LR_C_RANGE,
            "classifier__estimator__class_weight": Config.LR_CLASS_WEIGHTS,
        }

        # Configure Grid Search
        # We use n_jobs=1 for GridSearch because BaggingClassifier already uses n_jobs=-1
        # and we want to avoid oversubscribing threads.
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3 if debug else 5,  # Reduced internal CV for debug
            n_jobs=1,
            verbose=1,
        )

        logger.info("Starting Grid Search for hyperparameters...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Best CV AUC: {best_score:.6f}")
        logger.info(f"Best Parameters: {best_params}")

        # Predict on Validation Set
        val_probs = best_model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val, val_probs)
        logger.info(f"Fold {fold + 1} Validation AUC: {fold_auc:.10f}")

        # Save Model
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
        save_object(best_model, model_path)
        model_paths.append(model_path)
        logger.info(f"Saved model to {model_path}")

    # 4. Overall Evaluation
    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"\n{'='*40}")
    logger.info(f"Overall OOF AUC: {overall_auc:.10f}")
    logger.info(f"{'='*40}")

    return model_paths


def generate_submission(model_paths, debug=False):
    """
    Generates predictions for the test set using the ensemble of trained models.

    Args:
        model_paths (list): List of paths to saved model files.
        debug (bool): If True, runs on a small subset of test data.
    """
    logger.info("Starting submission generation...")

    # 1. Load Test Data
    preprocessor = FeaturePreprocessor()
    data = preprocessor.get_data(split="test", load_cached=True, debug=debug)

    X_test = data["X"]
    ids_test = data["ids"]

    # 2. Generate Predictions
    # We average the probabilities from all fold models (CV-Bagging)
    avg_probs = np.zeros(len(X_test))

    for path in model_paths:
        logger.info(f"Loading model from {path}...")
        model = load_object(path)

        probs = model.predict_proba(X_test)[:, 1]
        avg_probs += probs

    avg_probs /= len(model_paths)

    # 3. Create Submission DataFrame
    df_sub = pd.DataFrame(
        {"request_id": ids_test, "requester_received_pizza": avg_probs}
    )

    # 4. Save Submission
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_sub.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")

    # Validation check on shape
    logger.info(f"Submission shape: {df_sub.shape}")
    if not debug:
        # Load sample submission to verify row count
        try:
            sample = pd.read_csv(Config.SAMPLE_SUBMISSION)
            if len(df_sub) != len(sample):
                logger.warning(
                    f"Submission row count ({len(df_sub)}) differs from sample ({len(sample)})"
                )
        except Exception as e:
            logger.warning(f"Could not verify against sample submission: {e}")


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing
    # The prompt asks to only implement the module functions.
    pass
