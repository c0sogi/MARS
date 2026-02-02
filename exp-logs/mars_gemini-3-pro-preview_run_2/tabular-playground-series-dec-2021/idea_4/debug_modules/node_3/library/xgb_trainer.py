import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


def train_xgb_fold(X_train, y_train, X_val, y_val, params=None):
    """
    Trains an XGBoost model for a single fold.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (np.ndarray): Training labels (0-indexed).
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels (0-indexed).
        params (dict, optional): Hyperparameter overrides.

    Returns:
        xgb.XGBClassifier: The trained model.
    """
    # Load default parameters from Config
    xgb_params = Config.XGB_PARAMS.copy()

    # Update with any provided overrides
    if params:
        xgb_params.update(params)

    # Initialize the classifier
    # Note: In recent XGBoost versions, early_stopping_rounds and eval_metric
    # are accepted in the constructor.
    model = xgb.XGBClassifier(**xgb_params)

    # Fit the model
    # eval_set is required for early stopping to monitor validation performance
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Generate validation predictions
    # predict_proba returns (N_samples, N_classes)
    val_preds_proba = model.predict_proba(X_val)
    val_preds_cls = np.argmax(val_preds_proba, axis=1)

    # Calculate metrics
    # LogLoss requires probabilities
    loss = log_loss(y_val, val_preds_proba, labels=list(range(Config.NUM_CLASSES)))
    # Accuracy requires class labels
    acc = accuracy_score(y_val, val_preds_cls)

    # Print metrics with full precision as requested
    print(f"XGBoost Validation LogLoss: {loss}")
    print(f"XGBoost Validation Accuracy: {acc}")

    return model


def predict_xgb(model, X_test):
    """
    Generates predictions using a trained XGBoost model.

    Args:
        model (xgb.XGBClassifier): The trained model.
        X_test (pd.DataFrame or np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities (N_samples, N_classes).
    """
    return model.predict_proba(X_test)
