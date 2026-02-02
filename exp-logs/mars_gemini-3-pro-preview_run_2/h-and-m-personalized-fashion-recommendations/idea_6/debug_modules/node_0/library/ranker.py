import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from typing import List, Optional, Union, Tuple
from pathlib import Path

from library.config import LGBM_PARAMS, Paths
from library.utils import setup_logger, CacheManager

logger = setup_logger("ranker")


class Ranker:
    """
    Wraps LightGBM Ranker for the recommendation task.
    """

    def __init__(self, params: dict = LGBM_PARAMS):
        self.params = params.copy()
        self.model: Optional[lgb.Booster] = None
        self.cache = CacheManager()
        self.model_filename = "lgbm_ranker.txt"

    def _prepare_data(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "label"
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Sorts data by query (customer_id) and computes group counts for LGBM.
        """
        # Sort by customer_id to ensure contiguous groups
        # kind='mergesort' is stable
        df_sorted = df.sort_values("customer_id", kind="mergesort")

        # Compute groups (number of items per customer)
        # groupby(sort=False) preserves the order of appearance, which matches our sorted df
        group_counts = (
            df_sorted.groupby("customer_id", sort=False)["customer_id"].count().values
        )

        return df_sorted, group_counts, df_sorted[target_col]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "label",
        load_cached_model: bool = True,
    ) -> None:
        """
        Trains the LightGBM Ranker.

        Args:
            train_df: Training candidates with features and labels.
            val_df: Validation candidates with features and labels.
            feature_cols: List of feature column names.
            target_col: Name of the target column.
            load_cached_model: Whether to load a saved model if it exists.
        """
        # Check cache
        if load_cached_model and self.cache.exists(self.model_filename):
            logger.info(f"Loading cached model from {self.model_filename}...")
            self.model = lgb.Booster(
                model_file=str(self.cache.get_path(self.model_filename))
            )
            return

        logger.info("Preparing training data for LightGBM...")
        train_sorted, train_group, train_y = self._prepare_data(
            train_df, feature_cols, target_col
        )

        logger.info("Preparing validation data for LightGBM...")
        val_sorted, val_group, val_y = self._prepare_data(
            val_df, feature_cols, target_col
        )

        # Create Datasets
        train_set = lgb.Dataset(
            train_sorted[feature_cols],
            label=train_y,
            group=train_group,
            free_raw_data=False,
        )
        val_set = lgb.Dataset(
            val_sorted[feature_cols],
            label=val_y,
            group=val_group,
            reference=train_set,
            free_raw_data=False,
        )

        logger.info(f"Training LightGBM with params: {self.params}")

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 50)
            ),
            lgb.log_evaluation(period=10),
        ]

        # Train
        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log best score
        # Accessing best_score dict: {'valid': {'map@12': 0.XXXX}}
        if self.model.best_score:
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    logger.info(f"Best {dataset_name} {metric_name}: {score}")

        # Save model
        model_path = self.cache.get_path(self.model_filename)
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    def predict(self, test_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Scores the candidates in test_df.

        Args:
            test_df: DataFrame containing candidates and features.
            feature_cols: List of feature columns used for prediction.

        Returns:
            DataFrame with an additional 'score' column.
        """
        if self.model is None:
            # Try loading
            if self.cache.exists(self.model_filename):
                logger.info(f"Loading cached model from {self.model_filename}...")
                self.model = lgb.Booster(
                    model_file=str(self.cache.get_path(self.model_filename))
                )
            else:
                raise ValueError("Model not trained and no cache found.")

        logger.info(f"Predicting scores for {len(test_df)} candidates...")

        # Predict
        scores = self.model.predict(test_df[feature_cols])

        # Return result
        result_df = test_df.copy()
        result_df["score"] = scores
        return result_df

    def generate_submission(
        self, scored_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generates the submission file from scored candidates.
        Selects top 12 items per customer.

        Args:
            scored_df: DataFrame with 'customer_id', 'article_id', and 'score'.
            output_path: Path to save the CSV. If None, uses default from Paths.

        Returns:
            Submission DataFrame.
        """
        logger.info("Generating submission file...")

        # Sort by customer and score (descending)
        # We use mergesort for stability
        scored_df = scored_df.sort_values(
            ["customer_id", "score"], ascending=[True, False], kind="mergesort"
        )

        # Take top 12
        top_k = scored_df.groupby("customer_id").head(12)

        # Format predictions as space-separated string
        submission = (
            top_k.groupby("customer_id")["article_id"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        submission.columns = ["customer_id", "prediction"]

        if output_path is None:
            output_path = Paths.SUBMISSION_DIR / "submission.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")

        return submission
