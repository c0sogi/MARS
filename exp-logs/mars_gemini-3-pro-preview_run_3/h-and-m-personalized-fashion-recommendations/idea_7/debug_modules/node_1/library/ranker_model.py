import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config

# Set global seeds for reproducibility
np.random.seed(Config.SEED)


class LGBMRanker:
    """
    Wrapper for LightGBM Ranking model (LambdaRank).
    Handles training, validation, and submission generation.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.feature_cols = None

    def train(self, train_df, val_df):
        """
        Trains the LightGBM ranker using the provided training and validation sets.

        Args:
            train_df (pd.DataFrame): Training data with features and 'label'.
            val_df (pd.DataFrame): Validation data with features and 'label'.
        """
        print("Preparing data for LightGBM training...")

        # 1. Sort data by customer_id (required for group/query boundaries in LambdaRank)
        train_df = train_df.sort_values("customer_id").reset_index(drop=True)
        val_df = val_df.sort_values("customer_id").reset_index(drop=True)

        # 2. Define Feature Columns
        # Exclude metadata, IDs, and target columns
        ignore_cols = ["customer_id", "article_id", "label", "t_dat"]
        self.feature_cols = [c for c in train_df.columns if c not in ignore_cols]
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        # 3. Create Query Groups (number of items per customer)
        # This tells LightGBM which rows belong to the same query/user for ranking
        train_group = train_df.groupby("customer_id").size().values
        val_group = val_df.groupby("customer_id").size().values

        # 4. Create LightGBM Datasets
        train_set = lgb.Dataset(
            train_df[self.feature_cols],
            label=train_df["label"],
            group=train_group,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )

        val_set = lgb.Dataset(
            val_df[self.feature_cols],
            label=val_df["label"],
            group=val_group,
            reference=train_set,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )

        # 5. Train
        print("Starting LightGBM training...")
        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.params["early_stopping_rounds"]
                ),
                lgb.log_evaluation(period=50),
            ],
        )

        # 6. Save Model
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        model_path = Config.WORKING_DIR / "lgbm_model.txt"
        self.model.save_model(str(model_path))
        print(f"Model saved to {model_path}")

        # 7. Print Exact Metrics (Full Precision)
        if self.model.best_score:
            print("=" * 30)
            print("Best Validation Scores (Full Precision):")
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"{dataset_name} {metric_name}: {score}")
            print("=" * 30)

    def predict(self, test_df):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_df (pd.DataFrame): Candidate dataframe for test users containing features.
        """
        print("Generating predictions...")

        # 1. Load Model if not in memory
        if self.model is None:
            model_path = Config.WORKING_DIR / "lgbm_model.txt"
            if model_path.exists():
                print(f"Loading model from {model_path}...")
                self.model = lgb.Booster(model_file=str(model_path))
            else:
                raise ValueError("Model not trained or found at specified path.")

        # 2. Align Features
        # Get feature names stored in the model
        model_features = self.model.feature_name()

        # Verify columns exist in test data
        missing_cols = [c for c in model_features if c not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Missing features in test data: {missing_cols}")

        X_test = test_df[model_features]

        # 3. Predict Scores
        scores = self.model.predict(X_test)

        # Create a copy to avoid SettingWithCopyWarning
        test_df = test_df.copy()
        test_df["score"] = scores

        # 4. Rank and Select Top 12
        print("Sorting and selecting Top-12 items per customer...")
        # Sort by customer (asc) and score (desc)
        test_df = test_df.sort_values(["customer_id", "score"], ascending=[True, False])

        # Take top 12 items per customer
        top_12 = test_df.groupby("customer_id").head(12)

        # 5. Format for Submission
        # Convert article_id to string with leading zeros (e.g., 123 -> "0000000123")
        top_12["article_id_str"] = top_12["article_id"].astype(str).str.zfill(10)

        # Aggregate to space-separated string per customer
        submission_df = (
            top_12.groupby("customer_id")["article_id_str"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        submission_df.columns = ["customer_id", "prediction"]

        # 6. Merge with Sample Submission Template
        # This ensures we have rows for ALL customers in the test set, even if retrieval returned no candidates
        print("Merging with sample submission template...")
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        final_sub = sample_sub[["customer_id"]].merge(
            submission_df, on="customer_id", how="left"
        )

        # Fill missing predictions with empty string (or fallback if desired, but empty is safe)
        final_sub["prediction"] = final_sub["prediction"].fillna("")

        # 7. Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
