import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import compute_mcc, seed_everything


class DualStreamPredictor:
    """
    Manages two independent XGBoost models:
    - Stream A: Player-Player interactions
    - Stream B: Player-Ground impacts

    Handles training, threshold optimization, and inference for each stream.
    """

    def __init__(self):
        self.models = {}
        self.thresholds = {}

        # Define paths for persistence
        self.model_paths = {
            "A": os.path.join(Config.WORKING_DIR, "model_streamA.json"),
            "B": os.path.join(Config.WORKING_DIR, "model_streamB.json"),
        }
        self.threshold_paths = {
            "A": os.path.join(Config.WORKING_DIR, "threshold_streamA.joblib"),
            "B": os.path.join(Config.WORKING_DIR, "threshold_streamB.joblib"),
        }

    def _get_params(self):
        """Returns a copy of the XGBoost parameters from Config."""
        return Config.XGB_PARAMS.copy()

    def train_stream(self, X_train, y_train, X_val, y_val, stream_type):
        """
        Trains an XGBoost model for the specified stream, optimizes the decision threshold
        based on validation MCC, and saves the artifacts.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (np.ndarray): Validation labels.
            stream_type (str): "A" or "B".
        """
        print(f"Training Stream {stream_type} Model...")

        # Initialize model with parameters from Config
        params = self._get_params()
        clf = xgb.XGBClassifier(**params)

        # Fit the model
        # Note: early_stopping_rounds is in params (constructor),
        # so we just pass eval_set here.
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predict probabilities on validation set
        val_probs = clf.predict_proba(X_val)[:, 1]

        # Calculate and print LogLoss
        ll = log_loss(y_val, val_probs, labels=[0, 1])
        print(f"Stream {stream_type} Validation LogLoss: {ll}")

        # Optimize Threshold for MCC
        print(f"Optimizing threshold for Stream {stream_type}...")
        best_threshold = 0.5
        best_mcc = -1.0

        # Linear search for best threshold
        thresholds = np.linspace(0.01, 0.99, 99)
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            mcc = compute_mcc(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        print(f"Stream {stream_type} Best Threshold: {best_threshold}")
        print(f"Stream {stream_type} Best MCC: {best_mcc}")

        # Update internal state
        self.models[stream_type] = clf
        self.thresholds[stream_type] = best_threshold

        # Save artifacts
        clf.save_model(self.model_paths[stream_type])
        joblib.dump(best_threshold, self.threshold_paths[stream_type])

        return clf, best_threshold

    def predict(self, X, stream_type):
        """
        Generates binary predictions for the given input features using the
        specified stream's model and optimized threshold.

        Args:
            X (pd.DataFrame): Input features.
            stream_type (str): "A" or "B".

        Returns:
            np.ndarray: Binary predictions (0 or 1).
        """
        # Load model if not in memory
        if stream_type not in self.models:
            if os.path.exists(self.model_paths[stream_type]):
                clf = xgb.XGBClassifier()
                clf.load_model(self.model_paths[stream_type])
                self.models[stream_type] = clf
            else:
                raise FileNotFoundError(
                    f"Model for stream {stream_type} not found. Please train it first."
                )

        # Load threshold if not in memory
        if stream_type not in self.thresholds:
            if os.path.exists(self.threshold_paths[stream_type]):
                self.thresholds[stream_type] = joblib.load(
                    self.threshold_paths[stream_type]
                )
            else:
                # Fallback if no threshold saved (unlikely if trained)
                self.thresholds[stream_type] = 0.5

        clf = self.models[stream_type]
        threshold = self.thresholds[stream_type]

        # Generate probabilities
        probs = clf.predict_proba(X)[:, 1]

        # Apply threshold
        preds = (probs >= threshold).astype(int)

        return preds
