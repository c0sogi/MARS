import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.utils import compute_qwk, post_process_preds


class GradientBoostingRegressor:
    """
    A wrapper for LightGBM Regressor.
    Cite solution_lesson_node_00004
    """

    def __init__(self):
        """
        Initializes the regressor with hyperparameters from Config.
        """
        self.params = Config.LGBM_PARAMS
        self.model = None
        self.model_path = Config.MODEL_PATH

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target scores.
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation target scores.
        """
        print("Initializing LightGBM training...")

        # Create dataset objects
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=50),
            ],
        )

        print("Training complete.")

        # --- Evaluation ---

        # Training Metrics
        train_preds_raw = self.model.predict(X_train)
        train_preds_int = post_process_preds(train_preds_raw)
        train_qwk = compute_qwk(y_train, train_preds_int)
        train_mse = mean_squared_error(y_train, train_preds_raw)

        print("=== Training Metrics ===")
        print(f"Train MSE: {train_mse}")
        print(f"Train QWK: {train_qwk}")

        # Validation Metrics
        if X_val is not None and y_val is not None:
            val_preds_raw = self.model.predict(X_val)
            val_preds_int = post_process_preds(val_preds_raw)
            val_qwk = compute_qwk(y_val, val_preds_int)
            val_mse = mean_squared_error(y_val, val_preds_raw)

            print("=== Validation Metrics ===")
            print(f"Validation MSE: {val_mse}")
            print(f"Validation QWK: {val_qwk}")

        # Save model
        self.save_model()

    def predict(self, X):
        """
        Generates predictions for the given input features.
        """
        if self.model is None:
            self.load_model()

        return self.model.predict(X)

    def save_model(self):
        """
        Saves the LightGBM model to a text file.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(self.model_path)

    def load_model(self):
        """
        Loads the LightGBM model from the text file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No saved model found at {self.model_path}. Please train first."
            )

        print(f"Loading model from {self.model_path}...")
        self.model = lgb.Booster(model_file=self.model_path)
