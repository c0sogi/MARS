import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed


class LGBMRanker:
    """
    Encapsulates the LightGBM Regressor for ranking markdown cells.
    """

    def __init__(self):
        """
        Initializes the ranker with configuration parameters.
        """
        set_seed(Config.SEED)
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.features = [
            "n_code",
            "md_len",
            "smoothed_best_match_loc",
            "smoothed_center_of_mass",
            "sim_max",
        ]
        self.target = "rank"

    def train(self, train_df, val_df):
        """
        Trains the LightGBM model using the provided training and validation data.

        Args:
            train_df (pd.DataFrame): DataFrame containing training features and target.
            val_df (pd.DataFrame): DataFrame containing validation features and target.
        """
        print(
            f"Training LGBMRanker with {len(train_df)} training samples and {len(val_df)} validation samples."
        )

        # Prepare feature matrices and target vectors
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        X_val = val_df[self.features]
        y_val = val_df[self.target]

        # Create LightGBM datasets
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Define callbacks for early stopping and logging
        # stopping_rounds=50: Stop if validation metric doesn't improve for 50 rounds
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(
                period=0
            ),  # Suppress internal logging to keep output clean
        ]

        # Train the model
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            num_boost_round=self.params.get("n_estimators", 1000),
            callbacks=callbacks,
        )

        # Save the model
        os.makedirs(os.path.dirname(Config.LGBM_MODEL_PATH), exist_ok=True)
        self.model.save_model(Config.LGBM_MODEL_PATH)
        print(f"Model saved to {Config.LGBM_MODEL_PATH}")

        # Evaluate and print full precision metric
        # best_score is a dict: {'train': {'rmse': ...}, 'valid': {'rmse': ...}}
        val_rmse = self.model.best_score["valid"]["rmse"]
        print(f"Final Validation RMSE: {val_rmse}")

    def predict(self, test_df):
        """
        Generates predictions for the test dataset.

        Args:
            test_df (pd.DataFrame): DataFrame containing test features.

        Returns:
            np.ndarray: Array of predicted normalized ranks.
        """
        # Load model if not present in memory
        if self.model is None:
            if os.path.exists(Config.LGBM_MODEL_PATH):
                print(f"Loading model from {Config.LGBM_MODEL_PATH}")
                self.model = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)
            else:
                raise FileNotFoundError(
                    f"Model file not found at {Config.LGBM_MODEL_PATH}. Please train first."
                )

        # Ensure features exist
        missing_cols = [col for col in self.features if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Test DataFrame is missing columns: {missing_cols}")

        X_test = test_df[self.features]

        # Predict
        predictions = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )

        return predictions
