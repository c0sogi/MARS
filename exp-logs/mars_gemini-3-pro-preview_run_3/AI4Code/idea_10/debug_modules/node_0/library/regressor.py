import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed


class RankRegressor:
    """
    Wraps the LightGBM regression model for the Distribution-Aware Semantic Regressor.
    Predicts the normalized position (rank) of markdown cells based on semantic
    and structural features.
    """

    def __init__(self):
        """
        Initializes the regressor with configuration parameters.
        """
        self.params = Config.LGBM_PARAMS.copy()
        self.model_path = Config.LGBM_MODEL_PATH
        self.model = None
        set_seed(Config.SEED)

    def train(self, df_train, df_val):
        """
        Trains the LightGBM model using the provided training and validation data.

        Args:
            df_train (pd.DataFrame): Training features and targets.
            df_val (pd.DataFrame): Validation features and targets.
        """
        print("Preparing data for LightGBM training...")

        # Identify feature columns: all columns except identifiers and target
        exclude_cols = {"id", "cell_id", "target"}
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        print(f"Training with {len(feature_cols)} features: {feature_cols}")

        X_train = df_train[feature_cols]
        y_train = df_train["target"]

        X_val = df_val[feature_cols]
        y_val = df_val["target"]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        # Note: Using callbacks for early stopping as per LightGBM 4.x standards
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress periodic logging
        ]

        print("Starting LightGBM training...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,  # High number, controlled by early stopping
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

        # Evaluation
        y_pred_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        mse = np.mean((y_val - y_pred_val) ** 2)
        rmse = np.sqrt(mse)

        print(f"Validation RMSE: {rmse}")

    def predict(self, df_test):
        """
        Generates predictions for the test dataset.

        Args:
            df_test (pd.DataFrame): Test features.

        Returns:
            np.array: Predicted normalized ranks.
        """
        # Load model if not currently loaded
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading LightGBM model from {self.model_path}")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. Please train first."
                )

        # Identify feature columns
        exclude_cols = {"id", "cell_id", "target"}
        feature_cols = [c for c in df_test.columns if c not in exclude_cols]

        # Ensure features match those used in training (basic check)
        # LightGBM handles feature name matching, but we ensure correct subset
        X_test = df_test[feature_cols]

        predictions = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )

        return predictions
