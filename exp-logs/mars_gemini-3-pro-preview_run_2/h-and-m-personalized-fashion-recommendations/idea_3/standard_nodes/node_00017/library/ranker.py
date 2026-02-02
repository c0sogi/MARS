import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from library.config import Config
from library.utils import Timer


class Ranker:
    """
    Wraps LightGBM training and prediction logic for the Recommender System.
    Implements training with early stopping and submission generation.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS
        self.working_dir = Config.WORKING_DIR
        self.model = None

    def train(self, train_df, val_df, feature_cols):
        """
        Trains the LightGBM ranker.

        Args:
            train_df (pd.DataFrame): Training data with features and labels.
            val_df (pd.DataFrame): Validation data with features and labels.
            feature_cols (list): List of feature column names to use.
        """
        print("[Ranker] Preparing datasets for training...")

        # Sort by customer_id_idx to ensure correct grouping for ranking tasks
        # This is required for creating the 'group' parameter in LightGBM
        with Timer("Sorting Train Data"):
            train_df = train_df.sort_values("customer_id_idx").reset_index(drop=True)

        with Timer("Sorting Val Data"):
            val_df = val_df.sort_values("customer_id_idx").reset_index(drop=True)

        # Prepare X and y
        X_train = train_df[feature_cols]
        y_train = train_df["label"]

        X_val = val_df[feature_cols]
        y_val = val_df["label"]

        # Calculate groups (number of records per customer)
        # Since data is sorted by customer_id_idx, we can just count occurrences
        # sort=False ensures we keep the order consistent with the dataframe
        group_train = train_df.groupby("customer_id_idx", sort=False).size().values
        group_val = val_df.groupby("customer_id_idx", sort=False).size().values

        # Create LightGBM Datasets
        train_set = lgb.Dataset(
            X_train, label=y_train, group=group_train, feature_name=feature_cols
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            group=group_val,
            reference=train_set,
            feature_name=feature_cols,
        )

        print(f"[Ranker] Training LightGBM with params: {self.params}")

        # Define callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        # Train
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.params.get("n_estimators", 1000),
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        model_path = self.working_dir / "lgbm_model.txt"
        self.model.save_model(str(model_path))
        print(f"[Ranker] Model saved to {model_path}")

        # Print Feature Importance
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        print("\n[Ranker] Top 10 Features by Gain:")
        print(importance.head(10))

    def predict(self, test_df, feature_cols):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_df (pd.DataFrame): Test candidates with features.
            feature_cols (list): List of feature columns used for training.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        if self.model is None:
            # Try loading from disk if not in memory
            model_path = self.working_dir / "lgbm_model.txt"
            if model_path.exists():
                print(f"[Ranker] Loading model from {model_path}")
                self.model = lgb.Booster(model_file=str(model_path))
            else:
                raise ValueError("Model not trained and no model file found.")

        print(f"[Ranker] Predicting on {len(test_df)} test candidates...")

        # Predict scores
        scores = self.model.predict(test_df[feature_cols])
        test_df["score"] = scores

        print("[Ranker] Selecting Top 12 candidates per customer...")

        # Sort by customer and score (descending)
        # We use mergesort for stability, though quicksort is faster
        test_df = test_df.sort_values(
            ["customer_id_idx", "score"], ascending=[True, False]
        )

        # Select top 12
        top_12 = test_df.groupby("customer_id_idx").head(12)

        # Clean up memory
        del test_df

        print("[Ranker] Mapping IDs to strings for submission...")

        # Load ID Maps
        article_map = pd.read_parquet(Config.PATH_ARTICLE_MAP)
        customer_map = pd.read_parquet(Config.PATH_CUSTOMER_MAP)

        # 1. Map Article Indices to Strings
        # top_12 has [customer_id_idx, article_id_idx]
        top_12 = top_12.merge(article_map, on="article_id_idx", how="left")

        # 2. Group by Customer Index and create prediction string
        # This is more memory efficient than merging customer strings first
        preds_series = top_12.groupby("customer_id_idx")["article_id"].apply(
            lambda x: " ".join(x)
        )

        # 3. Convert to DataFrame and Map Customer Indices to Strings
        submission = preds_series.reset_index()
        submission.columns = ["customer_id_idx", "prediction"]

        submission = submission.merge(customer_map, on="customer_id_idx", how="left")

        # Final Format
        submission = submission[["customer_id", "prediction"]]

        # Save
        print(f"[Ranker] Saving submission to {Config.PATH_SUBMISSION}")
        submission.to_csv(Config.PATH_SUBMISSION, index=False)

        return submission
