import os
import joblib
import lightgbm as lgb
import pandas as pd
from library import config


class FarePredictor:
    """
    A wrapper class for the LightGBM Regressor to predict taxi fares.
    Encapsulates configuration, training, prediction, and persistence logic.
    """

    def __init__(self):
        # Load parameters from config
        self.params = config.MODEL_PARAMS.copy()
        self.train_config = config.TRAIN_CONFIG.copy()

        # Map num_boost_round from TRAIN_CONFIG to n_estimators for LGBMRegressor
        if "num_boost_round" in self.train_config:
            self.params["n_estimators"] = self.train_config["num_boost_round"]

        # Define the exact features to be used for training and prediction
        self.features = (
            config.FEATURE_CONFIG["raw_features"]
            + config.FEATURE_CONFIG["generated_features"]
        )
        self.target_col = config.FEATURE_CONFIG["target_col"]

        # Initialize the LightGBM model
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, train_df, val_df):
        """
        Trains the model using the provided training and validation dataframes.
        Implements early stopping and metric logging.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
        """
        # Prepare feature matrices and target vectors
        X_train = train_df[self.features]
        y_train = train_df[self.target_col]

        X_val = val_df[self.features]
        y_val = val_df[self.target_col]

        # Setup callbacks for early stopping and logging
        callbacks = []

        if "early_stopping_rounds" in self.train_config:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.train_config["early_stopping_rounds"]
                )
            )

        if "verbose_eval" in self.train_config:
            callbacks.append(
                lgb.log_evaluation(period=self.train_config["verbose_eval"])
            )

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=self.params.get("metric", "rmse"),
            callbacks=callbacks,
        )

        # Print the best validation score with full precision
        if hasattr(self.model, "best_score_") and self.model.best_score_:
            # best_score_ structure is {'valid_0': {'rmse': 0.12345}}
            for dataset_key, metrics in self.model.best_score_.items():
                for metric_name, score in metrics.items():
                    print(
                        f"Final best score for {dataset_key} - {metric_name}: {score}"
                    )

    def predict(self, df):
        """
        Generates predictions for the given dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing features.

        Returns:
            np.ndarray: Array of predicted fare amounts.
        """
        # Ensure we only use the features the model was trained on
        X = df[self.features]
        return self.model.predict(X)

    def save_model(self, path):
        """
        Saves the trained model to the specified path using joblib.

        Args:
            path (str): Destination file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads a trained model from the specified path.

        Args:
            path (str): Source file path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
