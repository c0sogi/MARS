import numpy as np
from sklearn.linear_model import Ridge
from library.config import Config
from library.utils import get_logger

logger = get_logger("stacking")


def train_meta_learner(oof_preds_dict, y_true):
    """
    Trains a Ridge Regression meta-learner on the stacked OOF predictions.

    Args:
        oof_preds_dict (dict): Dictionary mapping model names to their OOF prediction arrays.
                               Format: { 'model_name': np.array([pred1, pred2, ...]) }
        y_true (np.ndarray): Ground truth binary labels.

    Returns:
        sklearn.linear_model.Ridge: The trained meta-model.
    """
    logger.info("Training Meta-Learner (Ridge Regression)...")

    # Sort keys to ensure consistent column order between training and inference
    model_names = sorted(oof_preds_dict.keys())
    logger.info(f"Stacking models in order: {model_names}")

    # Create feature matrix: (n_samples, n_models)
    # We stack the 1D prediction arrays as columns
    X_meta = np.column_stack([oof_preds_dict[name] for name in model_names])

    # Initialize Ridge Regression with Config parameters
    # Alpha controls L2 regularization strength
    meta_model = Ridge(alpha=Config.meta_alpha, random_state=Config.seed)

    # Fit the model
    meta_model.fit(X_meta, y_true)

    # Log model parameters for inspection
    logger.info(f"Meta-Learner Coefficients: {meta_model.coef_}")
    logger.info(f"Meta-Learner Intercept: {meta_model.intercept_}")

    return meta_model


def predict_meta_learner(meta_model, test_preds_dict):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        meta_model (sklearn.linear_model.Ridge): The trained Ridge model.
        test_preds_dict (dict): Dictionary mapping model names to their Test prediction arrays.

    Returns:
        np.ndarray: Final ensemble predictions clipped to the range [0, 1].
    """
    # Sort keys to ensure consistent column order matching the training phase
    model_names = sorted(test_preds_dict.keys())

    # Create feature matrix: (n_samples, n_models)
    X_test_meta = np.column_stack([test_preds_dict[name] for name in model_names])

    # Generate predictions
    final_preds = meta_model.predict(X_test_meta)

    # Clip probabilities to valid range [0, 1]
    # Linear models can sometimes output values slightly outside this range
    final_preds = np.clip(final_preds, 0.0, 1.0)

    return final_preds
