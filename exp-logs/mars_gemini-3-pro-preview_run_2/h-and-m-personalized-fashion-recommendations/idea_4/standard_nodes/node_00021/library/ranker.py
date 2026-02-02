import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import gc
from library.config import Config


class LGBMRanker:
    """
    Wraps LightGBM for Learning-to-Rank tasks.
    Handles data formatting (groups), training with early stopping, and submission generation.
    """

    def __init__(self):
        self.model = None
        self.feature_cols = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list = None,
    ):
        """
        Trains the LightGBM ranker.

        Args:
            train_df: Training data with features and 'target' column.
            val_df: Validation data with features and 'target' column.
            feature_cols: List of feature column names to use. If None, infers from df.
        """
        print("Preparing data for LightGBM training...")

        # 1. Feature Selection
        if feature_cols is None:
            # Exclude non-feature columns
            exclude_cols = [
                "customer_id",
                "article_id",
                "t_dat",
                "target",
                "sales_channel_id",  # Often categorical/metadata
                "image_path",
            ]
            self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        else:
            self.feature_cols = feature_cols

        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        # 2. Sort Data for Grouping
        # LightGBM ranker requires queries (users) to be contiguous
        train_df = train_df.sort_values("customer_id").reset_index(drop=True)
        val_df = val_df.sort_values("customer_id").reset_index(drop=True)

        # 3. Create LGBM Datasets
        # Training Set
        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        # Compute query group sizes
        train_groups = train_df.groupby("customer_id").size().values

        train_set = lgb.Dataset(
            X_train, label=y_train, group=train_groups, free_raw_data=False
        )

        # Validation Set
        X_val = val_df[self.feature_cols]
        y_val = val_df["target"]
        val_groups = val_df.groupby("customer_id").size().values

        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            group=val_groups,
            reference=train_set,
            free_raw_data=False,
        )

        # 4. Train
        print("Starting LightGBM training...")

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=10),
        ]

        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_set,
            num_boost_round=Config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 5. Save Model
        model_path = Config.WORKING_DIR / "lgbm_model.txt"
        print(f"Saving model to {model_path}")
        self.model.save_model(str(model_path))

        # Log final metric
        if self.model.best_score:
            val_score = self.model.best_score["valid"]["map@12"]
            print(f"Final Validation MAP@12: {val_score}")

        # Clean up
        del X_train, y_train, train_groups, train_set
        del X_val, y_val, val_groups, val_set
        gc.collect()

    def predict(
        self,
        test_df: pd.DataFrame,
        output_path: str = None,
        load_model: bool = True,
    ):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_df: Test candidates with features.
            output_path: Path to save the submission CSV. Defaults to Config.SUBMISSION_DIR.
            load_model: Whether to load the model from disk if not in memory.
        """
        # 1. Load Model
        if self.model is None and load_model:
            model_path = Config.WORKING_DIR / "lgbm_model.txt"
            if model_path.exists():
                print(f"Loading model from {model_path}")
                self.model = lgb.Booster(model_file=str(model_path))
            else:
                raise FileNotFoundError("No trained model found. Run fit() first.")

        if self.feature_cols is None or len(self.feature_cols) == 0:
            # Infer features again if lost (e.g. fresh instantiation)
            # Assuming test_df has the same structure as train minus target
            exclude_cols = [
                "customer_id",
                "article_id",
                "t_dat",
                "target",
                "sales_channel_id",
                "image_path",
            ]
            self.feature_cols = [c for c in test_df.columns if c not in exclude_cols]

        print("Predicting scores...")

        # 2. Predict
        # We process in chunks if necessary, but 220GB RAM is plenty for inference
        X_test = test_df[self.feature_cols]
        scores = self.model.predict(X_test)

        # Assign scores back to dataframe
        # We work on a copy of the ID columns to avoid modifying the input
        results = test_df[["customer_id", "article_id"]].copy()
        results["score"] = scores

        # 3. Rank and Format
        print("Ranking and formatting submission...")

        # Sort by customer and score
        results = results.sort_values(["customer_id", "score"], ascending=[True, False])

        # Take top 12
        top_results = results.groupby("customer_id").head(12)

        # Group into space-separated string
        submission = (
            top_results.groupby("customer_id")["article_id"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        submission.columns = ["customer_id", "prediction"]

        # 4. Save
        if output_path is None:
            output_path = Config.SUBMISSION_DIR / "submission.csv"

        print(f"Saving submission to {output_path}")
        submission.to_csv(output_path, index=False)

        return submission
