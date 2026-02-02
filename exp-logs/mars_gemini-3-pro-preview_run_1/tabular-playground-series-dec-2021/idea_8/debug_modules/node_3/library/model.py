import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from library.config import Config


class XGBModelWrapper:
    """
    Wrapper for XGBoost Classifier to handle training and inference
    consistent with the pipeline requirements.
    """

    def __init__(self, params=None):
        # Load default params if none provided
        self.params = params if params else Config.get_xgb_params()

        # Ensure we output probabilities for soft voting and mlogloss
        # multi:softmax outputs hard labels, multi:softprob outputs probabilities.
        # Both optimize the same Softmax Cross Entropy loss.
        if self.params.get("objective") == "multi:softmax":
            self.params["objective"] = "multi:softprob"

        # Inject early stopping rounds into constructor params (modern XGBoost API)
        # This ensures compatibility with recent versions and avoids deprecation warnings in .fit()
        if "early_stopping_rounds" not in self.params:
            self.params["early_stopping_rounds"] = Config.EARLY_STOPPING_ROUNDS

        # Initialize the XGBClassifier
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the model using the provided training and validation sets.
        Implements early stopping and logs validation metrics.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Define evaluation set for monitoring and early stopping
        eval_set = [(X_val, y_val)]

        # Train the model
        # verbose=Config.VERBOSE_EVAL controls the logging frequency
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=Config.VERBOSE_EVAL)

        # Generate predictions on validation set to log final performance
        # XGBClassifier automatically uses the best iteration if early stopping was triggered
        val_preds = self.model.predict(X_val)
        val_probs = self.model.predict_proba(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, val_preds)
        ll = log_loss(y_val, val_probs, labels=list(range(Config.NUM_CLASSES)))

        # Print metrics with full precision
        print(f"Validation Accuracy: {acc}")
        print(f"Validation Log Loss: {ll}")

    def predict(self, X):
        """
        Predicts class labels for the given input.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input.
        """
        return self.model.predict_proba(X)
