import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config
from library.utils import get_logger

logger = get_logger("model_xgb")


class XGBWrapper:
    """
    Wrapper class for XGBoost training and inference.
    Encapsulates model initialization, training with early stopping,
    saving/loading, and threshold optimization.
    """

    def __init__(self):
        """
        Initialize the XGBoost classifier with parameters from Config.
        """
        self.params = Config.XGB_PARAMS.copy()

        # Initialize the classifier
        # Note: We use the sklearn API (XGBClassifier) for compatibility
        # with standard fit/predict workflows and easy parameter management.
        self.model = xgb.XGBClassifier(**self.params)
        self.best_threshold = 0.5

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
        """
        logger.info(f"Starting XGBoost training with params: {self.params}")
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # Fit the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=Config.VERBOSE_EVAL,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
        )

        # Log best iteration and score
        if hasattr(self.model, "best_iteration"):
            best_iter = self.model.best_iteration
            logger.info(f"Best iteration: {best_iter}")

            # Retrieve evaluation results
            results = self.model.evals_result()
            # results structure: {'validation_0': {'logloss': [...]}, 'validation_1': {'logloss': [...]}}
            # validation_0 is train, validation_1 is val
            train_score = results["validation_0"]["logloss"][best_iter]
            val_score = results["validation_1"]["logloss"][best_iter]

            logger.info(f"Best Train LogLoss: {train_score}")
            logger.info(f"Best Validation LogLoss: {val_score}")

        # Save the model
        self.save_model()

    def predict(self, X):
        """
        Generates probability predictions for the positive class (contact).

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Predicted probabilities for class 1.
        """
        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, y_true, y_pred_proba):
        """
        Finds the best probability threshold to maximize MCC.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            float: The optimal threshold.
        """
        logger.info("Optimizing decision threshold based on MCC...")

        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_binary)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        logger.info(f"Optimal Threshold: {best_thresh}")
        logger.info(f"Best Validation MCC: {best_mcc}")

        self.best_threshold = best_thresh
        return best_thresh

    def save_model(self):
        """
        Saves the trained model to the path defined in Config.
        """
        model_path = Config.MODEL_PATH
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        logger.info(f"Saving model to {model_path}...")
        self.model.save_model(model_path)

    def load_model(self):
        """
        Loads a trained model from the path defined in Config.
        """
        model_path = Config.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info(f"Loading model from {model_path}...")
        self.model.load_model(model_path)
