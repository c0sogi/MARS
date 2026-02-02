import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import ModelConfig, PathConfig


class GradientBoostingRanker:
    """
    Wraps the LightGBM classifier for ranking long answer candidates.
    """

    def __init__(self):
        self.params = ModelConfig.LGBM_PARAMS
        self.booster = None
        self.model_path = PathConfig.MODEL_FILE

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Identifies feature columns in the dataframe.
        Assumes features start with 'f_'.
        """
        return [c for c in df.columns if c.startswith("f_")]

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the LightGBM model using the provided training and validation data.

        Args:
            train_df (pd.DataFrame): Training data with features and 'label'.
            val_df (pd.DataFrame): Validation data with features and 'label'.
        """
        if train_df is None or train_df.empty:
            print("Error: Training data is empty.")
            return

        if val_df is None or val_df.empty:
            print("Error: Validation data is empty.")
            return

        feature_cols = self._get_feature_columns(train_df)
        label_col = "label"

        print(f"Training on {len(train_df)} samples with {len(feature_cols)} features.")
        print(f"Validating on {len(val_df)} samples.")

        # Create LightGBM Datasets
        train_ds = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
        val_ds = lgb.Dataset(
            val_df[feature_cols], label=val_df[label_col], reference=train_ds
        )

        # Setup callbacks
        evals_result = {}
        callbacks = [
            lgb.log_evaluation(period=ModelConfig.VERBOSE_EVAL),
            lgb.early_stopping(stopping_rounds=ModelConfig.EARLY_STOPPING_ROUNDS),
            lgb.record_evaluation(evals_result),
        ]

        # Train
        self.booster = lgb.train(
            self.params,
            train_ds,
            num_boost_round=ModelConfig.NUM_BOOST_ROUND,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Print final validation metric with full precision
        if "valid" in evals_result and "binary_logloss" in evals_result["valid"]:
            # Get the score at the best iteration (or the last one if early stopping didn't trigger)
            # best_iteration is 0-based index if model has it, else len of history
            best_iter = self.booster.best_iteration
            # If best_iteration is 0 (default) or -1, it might mean use all or not set.
            # Usually evals_result stores all history. We want the min.
            min_loss = min(evals_result["valid"]["binary_logloss"])
            print(f"Best Validation Binary Logloss: {min_loss:.16f}")

        # Save the trained model
        self.save_model()

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates relevance probability scores for candidates in the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing feature columns.

        Returns:
            np.ndarray: Array of probability scores.
        """
        if self.booster is None:
            if not self.load_model():
                raise RuntimeError("Model not trained or loaded.")

        feature_cols = self._get_feature_columns(df)

        if not feature_cols:
            raise ValueError("No feature columns found in input dataframe.")

        # Predict
        # LightGBM predicts raw probabilities for binary classification by default
        scores = self.booster.predict(df[feature_cols])
        return scores

    def save_model(self):
        """
        Saves the trained booster to disk.
        """
        if self.booster is not None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.booster.save_model(self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print("No model to save.")

    def load_model(self) -> bool:
        """
        Loads the booster from disk.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_path):
            try:
                self.booster = lgb.Booster(model_file=self.model_path)
                print(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        else:
            print(f"Model file not found at {self.model_path}")
            return False
