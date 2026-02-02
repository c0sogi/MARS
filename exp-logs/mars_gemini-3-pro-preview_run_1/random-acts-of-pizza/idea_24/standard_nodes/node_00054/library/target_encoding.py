import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import (
    WORKING_DIR,
    HISTORY_COL,
    TARGET_COL,
    BAYESIAN_SMOOTHING_K,
    RANDOM_SEED,
)
from library.utils import ensure_directory


class BayesianTargetEncoder:
    """
    Implements Bayesian Target Encoding for user history (subreddits).
    Calculates smoothed probabilities of success (P(Pizza|Subreddit)) and aggregates them.
    """

    def __init__(self, k=BAYESIAN_SMOOTHING_K, n_folds=5):
        """
        Args:
            k (float): Smoothing parameter. Higher k pulls rare subreddits closer to global mean.
            n_folds (int): Number of folds for Stratified K-Fold in fit_transform_cv.
        """
        self.k = k
        self.n_folds = n_folds
        self.subreddit_stats = {}
        self.global_mean = 0.0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Computes global statistics (mean and subreddit-specific scores) on the provided DataFrame.
        This should be called on the full training set before using transform() on test/val sets.
        """
        # Explode the list column to have one row per subreddit-user pair
        exploded = df[[HISTORY_COL, TARGET_COL]].explode(HISTORY_COL)
        # Filter out NaN/None subreddits
        exploded = exploded[exploded[HISTORY_COL].notna()]

        # Calculate counts and means for each subreddit
        stats = exploded.groupby(HISTORY_COL)[TARGET_COL].agg(["count", "mean"])

        # Calculate global mean of the target
        self.global_mean = df[TARGET_COL].mean()

        # Bayesian Smoothing: (n * mean + k * global_mean) / (n + k)
        stats["score"] = (
            stats["count"] * stats["mean"] + self.k * self.global_mean
        ) / (stats["count"] + self.k)

        self.subreddit_stats = stats["score"].to_dict()
        self.is_fitted = True

    def transform(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Applies the learned statistics to the dataframe.
        Maps history subreddits to scores and computes Mean, Max, Min.
        """
        if not self.is_fitted:
            raise ValueError(
                "BayesianTargetEncoder must be fitted before calling transform."
            )

        cache_path = os.path.join(WORKING_DIR, f"bayes_history_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Bayesian encoding from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Computing Bayesian History Encoding for {split_name}...")

        feature_names = ["hist_mean_success", "hist_max_success", "hist_min_success"]
        output_df = pd.DataFrame(0.0, index=df.index, columns=feature_names)

        # Apply globally learned stats
        self._apply_scores(
            df.reset_index(drop=True),
            self.subreddit_stats,
            output_df,
            range(len(df)),
            self.global_mean,
        )

        # Ensure index matches original dataframe
        output_df.index = df.index

        ensure_directory(cache_path)
        output_df.to_parquet(cache_path)
        print(f"Saved Bayesian encoding to {cache_path}")
        return output_df

    def fit_transform_cv(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Performs Stratified K-Fold Target Encoding on the training set.
        Generates Out-Of-Fold (OOF) features to prevent data leakage.
        Also fits the global stats on the full dataset at the end.
        """
        cache_path = os.path.join(WORKING_DIR, f"bayes_history_{split_name}.parquet")

        # Even if we load cache, we should fit the global stats so the object is ready for inference
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Bayesian encoding from {cache_path}")
            self.fit(df)  # Ensure internal state is set for future transforms
            return pd.read_parquet(cache_path)

        print(f"Computing Bayesian History Encoding (CV) for {split_name}...")

        feature_names = ["hist_mean_success", "hist_max_success", "hist_min_success"]
        output_df = pd.DataFrame(0.0, index=df.index, columns=feature_names)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED
        )

        # Reset index to ensure alignment during CV loop
        df_reset = df.reset_index(drop=True)
        y = df_reset[TARGET_COL]

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_reset, y)):
            # Training subset for this fold
            fold_train = df_reset.iloc[train_idx]
            fold_val = df_reset.iloc[val_idx]

            # Calculate stats on fold_train
            fold_exploded = fold_train[[HISTORY_COL, TARGET_COL]].explode(HISTORY_COL)
            fold_exploded = fold_exploded[fold_exploded[HISTORY_COL].notna()]

            stats = fold_exploded.groupby(HISTORY_COL)[TARGET_COL].agg(
                ["count", "mean"]
            )
            fold_global_mean = fold_train[TARGET_COL].mean()

            # Compute scores for subreddits
            stats["score"] = (
                stats["count"] * stats["mean"] + self.k * fold_global_mean
            ) / (stats["count"] + self.k)
            score_map = stats["score"].to_dict()

            # Apply to fold_val (OOF predictions)
            self._apply_scores(
                fold_val, score_map, output_df, val_idx, fold_global_mean
            )

        # Handle index alignment if original df had non-range index
        output_df.index = df.index

        # Finally, fit on the whole dataset so the encoder is ready for test data
        self.fit(df)

        ensure_directory(cache_path)
        output_df.to_parquet(cache_path)
        print(f"Saved Bayesian encoding to {cache_path}")
        return output_df

    def _apply_scores(self, df_subset, score_map, output_df, indices, default_val):
        """
        Helper to map scores to history lists and aggregate.
        Updates output_df in place at the specified indices.
        """
        histories = df_subset[HISTORY_COL].tolist()

        means, maxs, mins = [], [], []

        for hist in histories:
            # Handle empty or malformed lists
            if not isinstance(hist, list) or len(hist) == 0:
                means.append(default_val)
                maxs.append(default_val)
                mins.append(default_val)
                continue

            # Map subreddits to scores
            scores = [score_map.get(sub, default_val) for sub in hist]

            if not scores:
                means.append(default_val)
                maxs.append(default_val)
                mins.append(default_val)
            else:
                means.append(np.mean(scores))
                maxs.append(np.max(scores))
                mins.append(np.min(scores))

        # Assign computed values to the output dataframe
        # We use iloc to assign to the specific rows corresponding to this fold/subset
        output_df.iloc[indices, 0] = means
        output_df.iloc[indices, 1] = maxs
        output_df.iloc[indices, 2] = mins
