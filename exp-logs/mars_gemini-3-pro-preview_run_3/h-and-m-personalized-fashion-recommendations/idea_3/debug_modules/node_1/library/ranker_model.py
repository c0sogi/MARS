import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from library import config
from library import utils


class LGBMRankerWrapper:
    """
    Wrapper for LightGBM Ranker (LambdaRank) to handle Stage 2 Re-Ranking.
    """

    def __init__(self):
        self.params = config.LGBM_PARAMS.copy()
        self.model = None
        self.feature_names = []

        # Define categorical features based on dataset analysis
        self.cat_features = [
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
            "club_member_status",
            "fashion_news_frequency",
            "sales_channel_id",
        ]

    def _prepare_data(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[List[int]]]:
        """
        Prepares data for LightGBM ranking.
        1. Sorts by customer_id (query).
        2. Extracts features and target.
        3. Computes group counts (items per query).
        """
        # Ensure data is sorted by query (customer_id) for LightGBM grouping
        # We also sort by retrieval rank/score implicitly if not shuffled,
        # but explicit sort by customer is required for 'group' construction.
        df_sorted = df.sort_values(by="customer_id", kind="mergesort")

        # Identify feature columns
        # Exclude IDs, targets, and metadata not used for prediction
        exclude_cols = [
            "customer_id",
            "article_id",
            "label",
            "t_dat",
            "image_path",
            "prediction",
        ]
        feature_cols = [c for c in df_sorted.columns if c not in exclude_cols]

        # Update feature names if this is the first run
        if not self.feature_names:
            self.feature_names = feature_cols

        X = df_sorted[feature_cols]

        y = None
        group = None

        if is_train:
            if "label" not in df_sorted.columns:
                raise ValueError("Label column missing for training data.")
            y = df_sorted["label"]

            # Compute group counts (number of rows per customer)
            # Since it's sorted, we can just count occurrences
            group = df_sorted.groupby("customer_id", sort=False).size().to_list()

        return X, y, group, df_sorted

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the LightGBM Ranker with Early Stopping.
        """
        utils.seed_everything(config.SEED)

        print(f"Preparing Training Data (Rows: {len(train_df)})...")
        X_train, y_train, group_train, _ = self._prepare_data(train_df, is_train=True)

        print(f"Preparing Validation Data (Rows: {len(val_df)})...")
        X_val, y_val, group_val, _ = self._prepare_data(val_df, is_train=True)

        # Create LightGBM Datasets
        # Note: We must specify categorical features here
        # Filter cat_features to only those present in X
        actual_cat_features = [c for c in self.cat_features if c in X_train.columns]

        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            group=group_train,
            feature_name=self.feature_names,
            categorical_feature=actual_cat_features,
            free_raw_data=False,
        )

        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            group=group_val,
            feature_name=self.feature_names,
            categorical_feature=actual_cat_features,
            free_raw_data=False,
            reference=train_set,
        )

        print(f"Training LightGBM with params: {self.params}")

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 50)
            ),
            lgb.log_evaluation(period=10),
        ]

        with utils.Timer("LightGBM Training"):
            self.model = lgb.train(
                self.params,
                train_set,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

        # Print Best Score with full precision
        if self.model.best_score:
            print("\nBest Iteration Metrics:")
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"{dataset_name} - {metric_name}: {score}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates scores for the provided dataframe.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Prepare X (no label needed)
        # Note: We don't strictly need to sort by customer_id for prediction unless we use group-based features,
        # but we keep consistency. However, for prediction, we just need X.
        # We use the internal prepare but ignore y and group.
        # Ensure we don't shuffle so we can map back to the input df.

        # We manually select features to ensure order matches training
        X = df[self.feature_names]

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importance dataframe.
        """
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values(by="importance", ascending=False)

    def generate_submission(self, test_df: pd.DataFrame, output_path: Path):
        """
        Generates the final submission file.
        1. Predicts scores for test candidates.
        2. Selects Top-12 per user.
        3. Formats and saves CSV.
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating submission.")

        print("Predicting scores for test set...")
        scores = self.predict(test_df)

        # Assign scores back to dataframe
        # We work on a copy to avoid modifying the input
        sub_df = test_df[["customer_id", "article_id"]].copy()
        sub_df["score"] = scores

        print("Selecting Top-12 recommendations per user...")
        # Sort by User and Score (descending)
        sub_df.sort_values(
            ["customer_id", "score"], ascending=[True, False], inplace=True
        )

        # Group and take head(12)
        # Using groupby().head() is efficient
        top_k_df = sub_df.groupby("customer_id").head(config.TOP_K_SUBMISSION)

        # Format article_id as string (leading zeros if necessary, though data usually handles this)
        # The competition format requires article_id as string 0xxxxxxxxx
        # Check current format
        if not pd.api.types.is_string_dtype(top_k_df["article_id"]):
            top_k_df["article_id"] = top_k_df["article_id"].astype(str).str.zfill(10)

        # Aggregate to space-separated string
        print("Aggregating predictions...")
        final_sub = (
            top_k_df.groupby("customer_id")["article_id"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        final_sub.columns = ["customer_id", "prediction"]

        # Ensure all customers from sample submission are present
        # Load sample submission to get full list of customers
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_CSV_PATH)
        all_customers = sample_sub[["customer_id"]]

        # Merge to include cold users (who might have been filtered out if no candidates were found)
        # If no candidates found (unlikely with fallback), prediction is NaN.
        # We fill NaN with empty string or a default fallback if desired.
        # (Though SparseRetriever usually guarantees candidates via popularity fallback)
        submission = all_customers.merge(final_sub, on="customer_id", how="left")

        # Fill NaNs (if any)
        # In a robust system, we might fill with global top 12.
        # Here we assume candidates covered everyone.
        missing_mask = submission["prediction"].isna()
        if missing_mask.any():
            print(
                f"Warning: {missing_mask.sum()} customers have no predictions. Filling with empty string."
            )
            submission.loc[missing_mask, "prediction"] = ""

        # Save
        print(f"Saving submission to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print("Submission saved successfully.")
