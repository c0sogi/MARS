import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import set_seed


class DualStreamTrainer:
    """
    Manages the training, threshold optimization, and inference for the
    Orthogonal-Physics Dual-Stream GBDT.
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.threshold_a = 0.5
        self.threshold_b = 0.5

        # Ensure reproducibility
        set_seed(Config.SEED)

    def train_xgboost(self, X_train, y_train, X_val, y_val, stream_type):
        """
        Configures and trains an XGBoost model for a specific stream.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.array): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (np.array): Validation labels.
            stream_type (str): 'A' for Interaction, 'B' for Impact.

        Returns:
            xgb.Booster: The trained model.
        """
        print(f"\n--- Training Stream {stream_type} Model ---")

        # Select hyperparameters based on stream type
        if stream_type == "A":
            params = Config.STREAM_A_PARAMS.copy()
        elif stream_type == "B":
            params = Config.STREAM_B_PARAMS.copy()
        else:
            raise ValueError("stream_type must be 'A' or 'B'")

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Watchlist for monitoring
        watchlist = [(dtrain, "train"), (dval, "eval")]

        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=watchlist,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=50,
        )

        # Store model
        if stream_type == "A":
            self.model_a = model
        else:
            self.model_b = model

        print(f"Stream {stream_type} training completed.")
        print(f"Best Iteration: {model.best_iteration}")
        print(f"Best Score (LogLoss): {model.best_score}")

        return model

    def optimize_threshold(self, y_true, y_pred_prob):
        """
        Performs a linear search to find the probability threshold that maximizes MCC.

        Args:
            y_true (np.array): Ground truth labels.
            y_pred_prob (np.array): Predicted probabilities.

        Returns:
            float: The optimal threshold.
            float: The maximum MCC score achieved.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Generate thresholds to test
        thresholds = np.linspace(0.01, 0.99, Config.THRESHOLD_OPT_STEPS)

        for thresh in thresholds:
            y_pred_binary = (y_pred_prob >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_binary)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        return best_threshold, best_mcc

    def fit(
        self,
        X_train_A,
        y_train_A,
        X_val_A,
        y_val_A,
        X_train_B,
        y_train_B,
        X_val_B,
        y_val_B,
    ):
        """
        Orchestrates the full training pipeline:
        1. Train Stream A
        2. Optimize Stream A Threshold
        3. Train Stream B
        4. Optimize Stream B Threshold
        """
        # --- Stream A ---
        self.train_xgboost(X_train_A, y_train_A, X_val_A, y_val_A, "A")

        # Generate validation predictions for optimization
        dval_a = xgb.DMatrix(X_val_A)
        val_probs_a = self.model_a.predict(dval_a)

        self.threshold_a, mcc_a = self.optimize_threshold(y_val_A, val_probs_a)
        print(f"Stream A Optimized Threshold: {self.threshold_a}")
        print(f"Stream A Validation MCC: {mcc_a}")

        # --- Stream B ---
        self.train_xgboost(X_train_B, y_train_B, X_val_B, y_val_B, "B")

        # Generate validation predictions for optimization
        dval_b = xgb.DMatrix(X_val_B)
        val_probs_b = self.model_b.predict(dval_b)

        self.threshold_b, mcc_b = self.optimize_threshold(y_val_B, val_probs_b)
        print(f"Stream B Optimized Threshold: {self.threshold_b}")
        print(f"Stream B Validation MCC: {mcc_b}")

    def predict(self, X, stream_type):
        """
        Generates probability predictions for a specific stream.

        Args:
            X (pd.DataFrame): Features.
            stream_type (str): 'A' or 'B'.

        Returns:
            np.array: Predicted probabilities.
        """
        model = self.model_a if stream_type == "A" else self.model_b

        if model is None:
            raise ValueError(
                f"Model for Stream {stream_type} has not been trained or loaded."
            )

        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

    def save_checkpoint(self, base_path=Config.WORKING_DIR):
        """
        Saves models and thresholds to disk.
        """
        os.makedirs(base_path, exist_ok=True)

        # Save Models
        if self.model_a:
            self.model_a.save_model(os.path.join(base_path, "model_stream_a.json"))
        if self.model_b:
            self.model_b.save_model(os.path.join(base_path, "model_stream_b.json"))

        # Save Thresholds
        thresholds = {"threshold_a": self.threshold_a, "threshold_b": self.threshold_b}
        joblib.dump(thresholds, os.path.join(base_path, "thresholds.joblib"))
        print(f"Checkpoint saved to {base_path}")

    def load_checkpoint(self, base_path=Config.WORKING_DIR):
        """
        Loads models and thresholds from disk.
        """
        path_a = os.path.join(base_path, "model_stream_a.json")
        path_b = os.path.join(base_path, "model_stream_b.json")
        path_thresh = os.path.join(base_path, "thresholds.joblib")

        if os.path.exists(path_a):
            self.model_a = xgb.Booster()
            self.model_a.load_model(path_a)
            print("Loaded Stream A model.")

        if os.path.exists(path_b):
            self.model_b = xgb.Booster()
            self.model_b.load_model(path_b)
            print("Loaded Stream B model.")

        if os.path.exists(path_thresh):
            thresholds = joblib.load(path_thresh)
            self.threshold_a = thresholds["threshold_a"]
            self.threshold_b = thresholds["threshold_b"]
            print(f"Loaded Thresholds: A={self.threshold_a}, B={self.threshold_b}")
