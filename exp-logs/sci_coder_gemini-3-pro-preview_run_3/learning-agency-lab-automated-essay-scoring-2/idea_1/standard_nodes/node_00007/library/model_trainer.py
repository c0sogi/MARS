import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.utils import compute_qwk, post_process_preds


class LGBMRegressor:
    """
    A wrapper for LightGBM Regressor.
    Cite solution_lesson_node_00004: Hybrid Embedding-Boosting Architectures
    """

    def __init__(self):
        """
        Initializes the regressor.
        """
        self.model = None
        self.model_path = os.path.join(Config.MODEL_DIR, "lgbm_model.txt")

        # Hyperparameters
        # Cite solution_lesson_node_00006: Mitigating Dominant Feature Bias in Hybrid Models
        # We use feature_fraction < 1.0 to ensure the model doesn't over-rely on
        # strong structural features (like length) and ignores embeddings.
        self.params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": Config.SEED,
            "n_jobs": -1,
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.
        """
        print("Initializing LightGBM training...")
        print(f"Hyperparameters: {self.params}")

        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mse", callbacks=callbacks
        )

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
        Generates predictions.
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
        # Save the booster directly
        self.model.booster_.save_model(self.model_path)

    def load_model(self):
        """
        Loads the LightGBM model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No saved model found at {self.model_path}. Please train first."
            )

        print(f"Loading model from {self.model_path}...")
        self.model = lgb.Booster(model_file=self.model_path)
