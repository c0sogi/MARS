import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import config
from library.utils import set_seed


class RankRegressor:
    """
    Wraps a LightGBM regressor to predict the normalized rank of markdown cells.
    """

    def __init__(self):
        """
        Initializes the regressor with parameters from config.
        """
        self.params = config.LGBM_PARAMS
        self.model = None
        # Features extracted by FeatureExtractor
        self.features = [
            "n_code",
            "md_len",
            "best_match_loc",
            "center_of_mass",
            "sim_max",
            "sim_mean",
        ]
        self.target = "target"

    def train(self, train_df, val_df):
        """
        Trains the LightGBM model using the provided training and validation data.
        Uses early stopping based on validation RMSE.

        Args:
            train_df (pd.DataFrame): Training data with features and target.
            val_df (pd.DataFrame): Validation data with features and target.
        """
        set_seed(config.SEED)

        # Prepare Feature Matrices and Target Vectors
        X_train = train_df[self.features]
        y_train = train_df[self.target]

        X_val = val_df[self.features]
        y_val = val_df[self.target]

        # Initialize the model
        self.model = lgb.LGBMRegressor(**self.params)

        # Define callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=0),  # Disable verbose logging
        ]

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # Calculate and print final validation metric
        val_preds = self.model.predict(X_val)
        mse = np.mean((y_val - val_preds) ** 2)
        rmse = np.sqrt(mse)
        print(f"Validation RMSE: {rmse}")

    def predict(self, df):
        """
        Generates predictions for the given data.

        Args:
            df (pd.DataFrame): Data containing the feature columns.

        Returns:
            np.ndarray: Predicted normalized ranks.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # Ensure input has the correct features
        X = df[self.features]
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the trained model to a file.

        Args:
            path (str): File path to save the model.
        """
        if self.model is None:
            print("No model to save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Handle saving for both LGBMRegressor wrapper and raw Booster
        if isinstance(self.model, lgb.LGBMRegressor):
            self.model.booster_.save_model(path)
        else:
            self.model.save_model(path)

        print(f"Model saved to {path}")

    def load(self, path):
        """
        Loads a model from a file.

        Args:
            path (str): File path to load the model from.
        """
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            return

        # Load as a Booster
        self.model = lgb.Booster(model_file=path)
        print(f"Model loaded from {path}")

    def get_feature_importance(self):
        """
        Retrieves feature importance from the trained model.

        Returns:
            dict: Mapping of feature names to their gain importance.
        """
        if self.model is None:
            return {}

        # Get booster depending on model type
        if isinstance(self.model, lgb.LGBMRegressor):
            booster = self.model.booster_
        else:
            booster = self.model

        importance = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()

        return dict(zip(feature_names, importance))
