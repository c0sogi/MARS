import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import RF_PARAMS, CACHE_DIR
from library.data_factory import DataBuilder
from library.utils import set_seed


class RFHandler:
    """
    Handler for the Random Forest stream of the solution.
    Manages model initialization, training, evaluation, and prediction.
    """

    def __init__(self, params=None):
        """
        Initialize the RFHandler.

        Args:
            params (dict, optional): Hyperparameters for RandomForestClassifier.
                                     Defaults to RF_PARAMS from config.
        """
        set_seed()
        self.params = params if params is not None else RF_PARAMS.copy()

        # Initialize the model
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model and evaluates on validation set if provided.

        Args:
            X_train (sparse matrix): Training features.
            y_train (array-like): Training labels.
            X_val (sparse matrix, optional): Validation features.
            y_val (array-like, optional): Validation labels.
        """
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Random Forest Validation ROC AUC: {val_auc}")
            return val_preds
        return None

    def predict(self, X_test):
        """
        Generates predictions for the test set.

        Args:
            X_test (sparse matrix): Test features.

        Returns:
            np.ndarray: Probability predictions for the positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        print("Generating Random Forest predictions...")
        # Return probabilities for class 1 (received pizza)
        return self.model.predict_proba(X_test)[:, 1]


def run_rf_pipeline(load_cached_data=True):
    """
    Orchestrates the Random Forest pipeline: data loading, training, and prediction.

    Args:
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        dict: A dictionary containing:
            - 'val_preds': Predictions on the validation set (if available).
            - 'test_preds': Predictions on the test set.
            - 'val_auc': The AUC score on the validation set.
    """
    # 1. Load Data
    print("Initializing DataBuilder for Random Forest...")
    data_builder = DataBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, _) = data_builder.get_rf_data()

    # 2. Initialize Handler
    handler = RFHandler()

    # 3. Train and Evaluate
    val_preds = handler.train(X_train, y_train, X_val, y_val)

    # Calculate metric for return
    val_auc = 0.0
    if val_preds is not None:
        val_auc = roc_auc_score(y_val, val_preds)

    # 4. Predict on Test
    test_preds = handler.predict(X_test)

    return {"val_preds": val_preds, "test_preds": test_preds, "val_auc": val_auc}
