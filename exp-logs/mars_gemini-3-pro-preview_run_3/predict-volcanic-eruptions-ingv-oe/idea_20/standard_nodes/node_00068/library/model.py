import os
import lightgbm as lgb
import numpy as np
from library.config import Config


class VolcanoLGBM:
    """
    Wrapper class for the LightGBM Regressor tailored for Volcano Eruption Prediction.
    Implements the 'Single, Highly-Optimized Regressor' strategy.
    """

    def __init__(self):
        # Load parameters from Config
        self.params = Config.MODEL_PARAMS.copy()
        self.train_params = Config.TRAIN_PARAMS.copy()
        self.model_path = Config.MODEL_OUTPUT_PATH
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model using the provided training and validation sets.
        Implements early stopping to prevent overfitting.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation target.

        Returns:
            lgb.Booster: The trained LightGBM model.
        """
        # Create LightGBM Datasets
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        # Extract training loop parameters
        # n_estimators is passed as num_boost_round to lgb.train
        num_boost_round = self.params.pop("n_estimators", 10000)
        early_stopping_rounds = self.train_params.get("early_stopping_rounds", 200)
        verbose_eval = self.train_params.get("verbose_eval", 100)

        # Configure Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=verbose_eval),
        ]

        print(
            f"Starting LightGBM training. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
        )

        # Train the model
        self.model = lgb.train(
            params=self.params,
            train_set=train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

        # Print Best Metrics with full precision
        if self.model.best_score:
            print("--- Best Iteration Metrics ---")
            for dataset, metrics in self.model.best_score.items():
                for metric, score in metrics.items():
                    print(f"{dataset} {metric}: {score}")

        return self.model

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix to predict on.

        Returns:
            np.ndarray: Predicted time_to_eruption values.
        """
        # Load model if not in memory
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise RuntimeError(
                    "Model has not been trained and no saved model file was found."
                )

        # Predict using the best iteration found during training
        return self.model.predict(X, num_iteration=self.model.best_iteration)
