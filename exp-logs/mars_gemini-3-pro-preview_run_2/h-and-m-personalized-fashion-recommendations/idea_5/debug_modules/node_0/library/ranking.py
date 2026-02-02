import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from pathlib import Path
from library import config
from library import utils


class LGBMRankerWrapper:
    """
    Wrapper for LightGBM Ranking model (LambdaRank).
    Handles data preparation, training, and submission generation.
    """

    def __init__(self):
        self.model = None
        self.model_path = config.WORKING_DIR / "lgbm_model.txt"

    def _add_labels(self, candidates_df, ground_truth_df):
        """
        Adds a binary 'label' column to candidates based on ground truth.
        Label = 1 if (customer_id, article_id) exists in ground_truth, else 0.
        """
        # Create a set of positive pairs for fast lookup or use merge
        # Merge is memory efficient enough for this scale if we drop duplicates first
        gt_pairs = ground_truth_df[["customer_id", "article_id"]].drop_duplicates()
        gt_pairs["label"] = 1

        # Left join candidates with ground truth
        labeled_df = candidates_df.merge(
            gt_pairs, on=["customer_id", "article_id"], how="left"
        )
        labeled_df["label"] = labeled_df["label"].fillna(0).astype(int)

        return labeled_df

    def _make_lgb_dataset(self, df, feature_cols, target_col="label"):
        """
        Converts DataFrame to lgb.Dataset with group info.
        Data must be sorted by customer_id before calling this.
        """
        # Calculate group sizes (number of candidates per query/customer)
        # Assuming df is already sorted by customer_id
        group_counts = df.groupby("customer_id", sort=False).size().values

        ds = lgb.Dataset(
            data=df[feature_cols],
            label=df[target_col],
            group=group_counts,
            feature_name=feature_cols,
            free_raw_data=False,
        )
        return ds

    def train(
        self,
        train_candidates,
        val_candidates,
        train_ground_truth,
        val_ground_truth,
        feature_cols,
        params=None,
        load_cached_model=False,
    ):
        """
        Trains the LightGBM Ranker.

        Args:
            train_candidates (pd.DataFrame): Features for training users.
            val_candidates (pd.DataFrame): Features for validation users.
            train_ground_truth (pd.DataFrame): Actual purchases for training users (validation period).
            val_ground_truth (pd.DataFrame): Actual purchases for validation users (validation period).
            feature_cols (list): List of feature column names.
            params (dict): LightGBM parameters.
            load_cached_model (bool): Whether to load model from disk.
        """
        # 1. Caching
        if load_cached_model and self.model_path.exists():
            print(f"Loading cached LightGBM model from {self.model_path}...")
            self.model = lgb.Booster(model_file=str(self.model_path))
            return

        print("Preparing data for LightGBM training...")

        if params is None:
            params = config.LGBM_PARAMS

        # 2. Labeling
        print("Labeling training data...")
        train_df = self._add_labels(train_candidates, train_ground_truth)

        print("Labeling validation data...")
        val_df = self._add_labels(val_candidates, val_ground_truth)

        # 3. Sorting (Crucial for LambdaRank grouping)
        print("Sorting data by customer_id...")
        train_df = train_df.sort_values("customer_id").reset_index(drop=True)
        val_df = val_df.sort_values("customer_id").reset_index(drop=True)

        # 4. Dataset Creation
        print("Creating LightGBM Datasets...")
        train_ds = self._make_lgb_dataset(train_df, feature_cols)
        val_ds = self._make_lgb_dataset(val_df, feature_cols)

        # Clean up DataFrames to free memory
        del train_df, val_df
        gc.collect()

        # 5. Training
        print(f"Starting training with params: {params}...")

        # Callbacks for logging
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50),
        ]

        self.model = lgb.train(
            params=params,
            train_set=train_ds,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 6. Save Model
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(str(self.model_path))

        # Feature Importance
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Features by Gain:")
        print(importance.head(10))

    def predict(self, test_candidates, feature_cols):
        """
        Generates predictions for test candidates and formats submission.

        Args:
            test_candidates (pd.DataFrame): DataFrame with features.
            feature_cols (list): List of feature columns used in training.

        Returns:
            pd.DataFrame: Submission dataframe with 'customer_id' and 'prediction'.
        """
        # Load model if needed
        if self.model is None:
            if self.model_path.exists():
                print(f"Loading LightGBM model from {self.model_path}...")
                self.model = lgb.Booster(model_file=str(self.model_path))
            else:
                raise ValueError("Model not trained and no cache found!")

        print("Predicting scores for test candidates...")

        # Make sure we don't modify original
        df = test_candidates.copy()

        # Predict
        # We process in chunks if necessary, but 220GB RAM is plenty for inference
        scores = self.model.predict(df[feature_cols])
        df["score"] = scores

        print("Ranking and selecting top 12 items...")

        # Sort by customer and score
        # We use a trick: sort values then groupby head
        df = df.sort_values(["customer_id", "score"], ascending=[True, False])

        # Select top 12
        top_12 = df.groupby("customer_id").head(12)

        # Aggregate to string
        print("Formatting submission string...")
        submission = (
            top_12.groupby("customer_id")["article_id"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        submission.columns = ["customer_id", "prediction"]

        # Verify submission format
        print(f"Generated predictions for {len(submission)} customers.")

        # Save
        submission_path = config.SUBMISSION_FILE
        print(f"Saving submission to {submission_path}...")
        submission.to_csv(submission_path, index=False)

        return submission
