import os
import numpy as np
import xgboost as xgb
from library.config import Config


class TaxiFareModel:
    """
    A gradient boosting model optimized for the NYC Taxi Fare prediction task.
    Wraps xgboost.XGBRegressor with physics-informed constraints.
    """

    def __init__(self):
        # Load hyperparameters from Config
        self.params = Config.get_model_params()
        self.model = xgb.XGBRegressor(**self.params)
        self.min_fare = Config.MIN_FARE

    def fit(self, X_train, y_train, X_val=None, y_val=None, patience=50):
        """
        Trains the model using XGBoost with early stopping.
        """
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            print("Training with validation set for early stopping...")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=patience,
            verbose=False,
        )

    def predict(self, X):
        """
        Predicts fare amounts for the given features.
        Enforces the minimum fare constraint.
        """
        predictions = self.model.predict(X)
        # Enforce lower bound (Physics/Domain constraint)
        return np.maximum(predictions, self.min_fare)

    def save(self, path):
        """
        Saves the model to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Ensure correct extension for XGBoost
        base, _ = os.path.splitext(path)
        save_path = base + ".json"
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

    def load(self, path):
        """
        Loads model from a file.
        """
        base, _ = os.path.splitext(path)
        load_path = base + ".json"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self.model.load_model(load_path)
        print(f"Model loaded from {load_path}")
