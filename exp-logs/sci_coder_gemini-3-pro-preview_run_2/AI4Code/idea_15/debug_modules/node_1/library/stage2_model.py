import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from typing import List, Optional

from library.config import Config


class LGBMRanker:
    """
    Stage 2: Multi-View Gradient Booster.
    Implements a LightGBM Regressor to refine cell ranking predictions by integrating
    Ridge predictions (Stage 1) with Uncertainty-Aware Multi-View Anchors.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "lgbm_model.txt")
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rank",
    ):
        """
        Trains the LightGBM model using the provided training and validation dataframes.
        Uses Early Stopping based on the validation set performance (MAE).

        Args:
            train_df: DataFrame containing training samples and features.
            val_df: DataFrame containing validation samples and features.
            feature_cols: List of column names to be used as features.
            target_col: Name of the target column (normalized rank).
        """
        # Prepare datasets
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Extract n_estimators for num_boost_round
        num_boost_round = self.params.pop("n_estimators", 5000)

        # Setup callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        print(
            f"Starting Stage 2 LightGBM Training with {len(feature_cols)} features..."
        )

        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        os.makedirs(self.working_dir, exist_ok=True)
        self.model.save_model(self.model_path)
        print(f"Stage 2 Model saved to {self.model_path}")

        # Validation Evaluation
        val_preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Stage 2 Validation MAE: {mae}")

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Generates rank predictions for the given dataframe.
        Loads the model from disk if it is not currently in memory.

        Args:
            df: DataFrame containing the samples to predict.
            feature_cols: List of feature columns to use for prediction.

        Returns:
            Numpy array of predicted normalized ranks.
        """
        # Load model if not present
        if self.model is None:
            if os.path.exists(self.model_path):
                # print(f"Loading Stage 2 model from {self.model_path}")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise FileNotFoundError(
                    f"LightGBM model not found at {self.model_path}. Call train() first."
                )

        # Predict
        return self.model.predict(df[feature_cols])
