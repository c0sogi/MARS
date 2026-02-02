import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config


class LGBMRanker:
    """
    Implements the Stage 2 LightGBM Ranker.
    This model consumes the Multi-View Neighborhood features and the Stage 1 Ridge predictions
    to produce the final normalized rank for markdown cells.
    """

    def __init__(self):
        """
        Initialize the ranker with configuration parameters.
        """
        self.params = Config.LGBM_PARAMS.copy()
        self.model_path = Config.CACHE_STAGE2_LGBM
        self.model = None

        # Extract the number of boosting rounds from params if present,
        # as lgb.train uses num_boost_round argument.
        self.num_boost_round = self.params.pop("n_estimators", 10000)

        # Define the features to be used for training/prediction
        # These must match the output of NeighborhoodFeatureExtractor
        self.feature_cols = [
            "ridge_pred",
            "lex_mean",
            "lex_wmean",
            "lex_std",
            "lex_min",
            "lex_max",
            "lat_mean",
            "lat_wmean",
            "lat_std",
            "lat_min",
            "lat_max",
            "n_code",
            "n_md",
            "code_ratio",
        ]

    def train_model(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        """
        Trains the LightGBM model using the provided training and validation sets.
        Implements early stopping and saves the model to disk.

        Args:
            df_train (pd.DataFrame): Training data containing features and 'pct_rank'.
            df_val (pd.DataFrame): Validation data containing features and 'pct_rank'.
        """
        print(
            f"Stage 2: Training LightGBM on {len(df_train)} samples (Val: {len(df_val)})..."
        )

        # Prepare Feature Matrices and Targets
        X_train = df_train[self.feature_cols]
        y_train = df_train["pct_rank"]

        X_val = df_val[self.feature_cols]
        y_val = df_val["pct_rank"]

        # Create LightGBM Datasets
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Define Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.LGBM_VERBOSE_EVAL),
        ]

        # Train
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        print(f"Stage 2 LightGBM model saved to {self.model_path}")

        # Print final validation metric manually for precision
        val_preds = self.model.predict(X_val)
        final_mae = np.mean(np.abs(y_val - val_preds))
        print(f"Final Validation MAE: {final_mae}")

    def predict_rank(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Generates rank predictions for the provided features using the trained model.
        Loads the model from disk if not already in memory.

        Args:
            df_features (pd.DataFrame): DataFrame containing the feature columns.

        Returns:
            np.ndarray: Predicted normalized ranks.
        """
        # Load model if needed
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 2 LightGBM model from {self.model_path}")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise FileNotFoundError(
                    f"Stage 2 model not found at {self.model_path}. Call train_model() first."
                )

        # Ensure features exist
        missing_cols = [c for c in self.feature_cols if c not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Missing features in input DataFrame: {missing_cols}")

        X = df_features[self.feature_cols]

        # Predict
        preds = self.model.predict(X)

        return preds
