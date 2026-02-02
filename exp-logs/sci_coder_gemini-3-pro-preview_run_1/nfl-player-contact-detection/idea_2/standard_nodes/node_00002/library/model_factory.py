import os
import numpy as np
import lightgbm as lgb
from library.config import Config


class LGBMClassifierWrapper:
    """
    A wrapper for the LightGBM Classifier to handle training with early stopping,
    class imbalance, and prediction.
    """

    def __init__(self):
        """
        Initialize the wrapper with parameters from Config.
        """
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.model_path = Config.MODEL_PATH

        # Extract n_estimators to use as num_boost_round in train()
        self.num_boost_round = self.params.pop("n_estimators", 2000)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model using the provided training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Define callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        print(f"Starting training with params: {self.params}")

        # Train the model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Save the trained model
        self.save_model()

    def predict(self, X):
        """
        Generates probability predictions for the input data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Probability scores for the positive class (contact).
        """
        if self.model is None:
            self.load_model()

        # LightGBM predict returns raw probabilities for binary classification
        return self.model.predict(X)

    def save_model(self):
        """
        Saves the model to disk.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print("No model to save.")

    def load_model(self):
        """
        Loads the model from disk.
        """
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train the model first."
            )
