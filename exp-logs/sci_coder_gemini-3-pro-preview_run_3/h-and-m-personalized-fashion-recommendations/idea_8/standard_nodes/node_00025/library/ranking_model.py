import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from library import config, feature_engine


class Ranker:
    """
    Wrapper for the LightGBM ranking model (Stage 2 of the cascade).
    Delegates complex data preparation and training loops to library.feature_engine.
    """

    def __init__(self, params=None):
        self.params = params if params else config.LGBM_PARAMS
        self.model = None

    def train(self, train_df, val_df):
        """
        Trains the ranker using the provided training and validation DataFrames.

        Args:
            train_df (pd.DataFrame): Training data with features and 'label'.
            val_df (pd.DataFrame): Validation data with features and 'label'.
        """
        # Delegate to feature_engine which handles Group/Query creation and lgb.Dataset formatting
        self.model = feature_engine.train_ranker(train_df, val_df, self.params)

        # Print best validation metrics with full precision
        if self.model.best_score:
            print("Best Validation Scores:")
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"{dataset_name} {metric_name}: {score}")

    def predict(self, test_df):
        """
        Generates raw scores for the provided candidates.

        Args:
            test_df (pd.DataFrame): Candidates with features.

        Returns:
            np.ndarray: Predicted scores.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # Filter columns to match those used in training
        # We exclude metadata columns to isolate features
        ignore_cols = [
            "customer_id",
            "article_id",
            "label",
            "t_dat",
            "prediction",
            "score",
        ]
        features = [c for c in test_df.columns if c not in ignore_cols]

        return self.model.predict(test_df[features])

    def generate_and_save_submission(self, test_df, sample_submission_df):
        """
        Generates the final submission file using the trained model.

        Args:
            test_df (pd.DataFrame): Test candidates with features.
            sample_submission_df (pd.DataFrame): Template for submission.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        feature_engine.generate_submission(self.model, test_df, sample_submission_df)

    def save(self, path):
        """
        Saves the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path):
        """
        Loads a trained model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = lgb.Booster(model_file=str(path))
