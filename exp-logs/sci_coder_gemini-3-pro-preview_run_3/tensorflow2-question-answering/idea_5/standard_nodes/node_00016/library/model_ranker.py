import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from library.configuration import Config
from library.data_loader import RankerDatasetBuilder


class GradientBoostingRanker:
    """
    Wrapper for a LightGBM model to rank candidate paragraphs for Long Answer selection.
    """

    def __init__(self):
        self.params = Config.RANKER_PARAMS
        self.model_path = Config.RANKER_MODEL_PATH
        self.model = None
        self.feature_cols = None

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding known metadata columns.
        """
        exclude_cols = {
            "example_id",
            "candidate_index",
            "start_token",
            "end_token",
            "label",
        }
        return [c for c in df.columns if c not in exclude_cols]

    def train(self, load_cached_data=True, sample_size=None):
        """
        Trains the LightGBM ranker model.

        Args:
            load_cached_data (bool): Whether to load pre-computed features from cache.
            sample_size (int, optional): Limit dataset size for debugging.
        """
        print("Preparing Ranker Training Data...")
        train_df = RankerDatasetBuilder.build_train_set(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        print("Preparing Ranker Validation Data...")
        val_df = RankerDatasetBuilder.build_val_set(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # Define feature columns
        self.feature_cols = self._get_feature_columns(train_df)
        print(f"Ranker Features ({len(self.feature_cols)}): {self.feature_cols}")

        # Prepare X and y
        X_train = train_df[self.feature_cols]
        y_train = train_df["label"]

        X_val = val_df[self.feature_cols]
        y_val = val_df["label"]

        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.RANKER_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ]

        print(f"Starting training with {len(X_train)} samples...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=Config.RANKER_NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log final metrics
        print("Ranker Training Finished.")
        if self.model.best_score:
            print("Best Validation Scores:")
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"{dataset_name} {metric_name}: {score}")

        self.save_model()

    def predict(self, features_df):
        """
        Predicts relevance scores for a set of candidates.

        Args:
            features_df (pd.DataFrame): DataFrame containing feature columns.

        Returns:
            np.array: Probability scores (0 to 1).
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            raise RuntimeError("Ranker model is not trained or loaded.")

        # Determine feature columns if not already set
        if self.feature_cols is None:
            self.feature_cols = self._get_feature_columns(features_df)

        # Ensure input has necessary columns
        missing_cols = set(self.feature_cols) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame missing features: {missing_cols}")

        X = features_df[self.feature_cols]

        # Predict
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        return preds

    def save_model(self):
        """Saves the LightGBM model to the configured path."""
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            print(f"Ranker model saved to {self.model_path}")

    def load_model(self):
        """Loads the LightGBM model from the configured path."""
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
            print(f"Ranker model loaded from {self.model_path}")
        else:
            print(f"No ranker model found at {self.model_path}")
