import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class HFBB:
    """
    Hierarchical Frequency Back-off (HFBB) Model.

    This class implements a 4-tier memory lookup system:
    1. Trigram: (prev, curr, next) -> target
    2. Bigram Prev: (prev, curr) -> target
    3. Bigram Next: (curr, next) -> target
    4. Unigram: (curr) -> target

    It uses strict priority: if a higher-order N-gram match is found, it is used.
    """

    def __init__(self):
        self.logger = setup_logger("HFBB")
        self.trigram_map = None
        self.bigram_prev_map = None
        self.bigram_next_map = None
        self.unigram_map = None

        # Padding tokens for context
        self.PAD_START = "^"
        self.PAD_END = "$"

    def _prepare_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to generate 'prev' and 'next' context columns respecting sentence boundaries.
        """
        # Ensure data is sorted to correctly shift
        # We assume df is already sorted by sentence_id, token_id, but explicit sort is safer
        # However, for massive datasets, sorting can be expensive. We rely on the provided order
        # or the user ensuring order. The metadata/train.csv is usually ordered.

        # Create copies to avoid SettingWithCopy warnings on input df
        df = df.copy()

        # Ensure string types
        df["before"] = df["before"].astype(str)

        # Shift to get context
        # We group by sentence_id to ensure we don't bleed context across sentences
        # Using shift within groups is cleaner but slower.
        # Faster approach: global shift + mask.

        # Global shift
        df["prev"] = df["before"].shift(1).fillna(self.PAD_START)
        df["next"] = df["before"].shift(-1).fillna(self.PAD_END)

        # Mask boundaries
        # If sentence_id changed from previous row, then 'prev' is invalid (start of new sentence)
        # If sentence_id changes to next row, then 'next' is invalid (end of current sentence)

        # Check sentence boundaries
        # Note: sentence_id can be int or str.
        sent_ids = df["sentence_id"].values

        # Mask Start: where current sent_id != prev sent_id
        # We need to handle the first row separately or use prepend
        is_start = np.concatenate(([True], sent_ids[1:] != sent_ids[:-1]))
        df.loc[is_start, "prev"] = self.PAD_START

        # Mask End: where current sent_id != next sent_id
        is_end = np.concatenate((sent_ids[:-1] != sent_ids[1:], [True]))
        df.loc[is_end, "next"] = self.PAD_END

        return df

    def _build_map(self, df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        """
        Builds a frequency map for a specific set of grouping columns.
        Returns a DataFrame with group_cols + ['after'] containing the most frequent mapping.
        """
        # Count occurrences of each mapping
        # group_cols + ['after']
        counts = df.groupby(group_cols + ["after"]).size().reset_index(name="count")

        # Sort by count descending to put most frequent first
        counts = counts.sort_values("count", ascending=False)

        # Drop duplicates keeping the first (most frequent)
        best_mapping = counts.drop_duplicates(subset=group_cols, keep="first")

        # Drop the count column, we just need the mapping
        return best_mapping.drop(columns=["count"])

    def fit(self, df: pd.DataFrame, load_cached_data: bool = True):
        """
        Builds the hierarchical frequency maps from the training data.

        Args:
            df (pd.DataFrame): Training data containing 'sentence_id', 'token_id', 'before', 'after'.
            load_cached_data (bool): If True, attempts to load maps from disk cache.
        """
        # Define cache paths
        cache_dir = Config.HFBB_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        paths = {
            "trigram": os.path.join(cache_dir, "trigram.parquet"),
            "bigram_prev": os.path.join(cache_dir, "bigram_prev.parquet"),
            "bigram_next": os.path.join(cache_dir, "bigram_next.parquet"),
            "unigram": os.path.join(cache_dir, "unigram.parquet"),
        }

        # Check if all exist
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            self.logger.info("Loading HFBB maps from cache...")
            self.trigram_map = pd.read_parquet(paths["trigram"])
            self.bigram_prev_map = pd.read_parquet(paths["bigram_prev"])
            self.bigram_next_map = pd.read_parquet(paths["bigram_next"])
            self.unigram_map = pd.read_parquet(paths["unigram"])
            self.logger.info("HFBB maps loaded.")
            return

        self.logger.info("Building HFBB maps from source data...")

        # Prepare context
        self.logger.info("Generating context columns...")
        df_ctx = self._prepare_context(df)
        df_ctx["after"] = df_ctx["after"].astype(str)

        # Build Maps
        self.logger.info("Building Trigram Map...")
        self.trigram_map = self._build_map(df_ctx, ["prev", "before", "next"])

        self.logger.info("Building Bigram Prev Map...")
        self.bigram_prev_map = self._build_map(df_ctx, ["prev", "before"])

        self.logger.info("Building Bigram Next Map...")
        self.bigram_next_map = self._build_map(df_ctx, ["before", "next"])

        self.logger.info("Building Unigram Map...")
        self.unigram_map = self._build_map(df_ctx, ["before"])

        # Save to cache
        self.logger.info("Saving maps to cache...")
        self.trigram_map.to_parquet(paths["trigram"], index=False)
        self.bigram_prev_map.to_parquet(paths["bigram_prev"], index=False)
        self.bigram_next_map.to_parquet(paths["bigram_next"], index=False)
        self.unigram_map.to_parquet(paths["unigram"], index=False)

        self.logger.info("HFBB training complete.")

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicts normalization for a batch of data using the hierarchical back-off strategy.

        Args:
            df (pd.DataFrame): Data containing 'sentence_id', 'token_id', 'before'.

        Returns:
            pd.Series: Predicted 'after' values. Entries are None/NaN if no match found in any tier.
        """
        if self.trigram_map is None:
            raise ValueError("HFBB model has not been fitted.")

        self.logger.info(f"Predicting batch of size {len(df)}...")

        # Prepare context
        df_ctx = self._prepare_context(df)

        # Initialize predictions with NaN
        # We use a separate series to track results
        results = pd.Series(index=df_ctx.index, data=np.nan, dtype=object)

        # Helper for merging
        # We perform left merge of the input data with the map.
        # The map has columns [keys..., 'after'].
        # We rename 'after' to 'pred_tier' to avoid collision if 'after' exists in input (e.g. during validation)

        def apply_tier(current_results, map_df, join_keys, tier_name):
            # Find indices that are still NaN
            missing_mask = current_results.isna()
            if not missing_mask.any():
                return current_results

            # Subset of data that needs prediction
            subset = df_ctx.loc[missing_mask, join_keys]

            # Merge
            merged = pd.merge(
                subset, map_df, on=join_keys, how="left", suffixes=("", "_map")
            )

            # The merge result aligns with 'subset' but we need to align with 'current_results'
            # We set the index of merged to match subset's index
            merged.index = subset.index

            # Update results
            # 'after' is the column name in the map_df
            found_preds = merged["after"]

            # Fill into main results
            # We only update where we found a prediction (not NaN in found_preds)
            update_mask = found_preds.notna()
            current_results.loc[update_mask.index[update_mask]] = found_preds[
                update_mask
            ]

            count = update_mask.sum()
            self.logger.info(f"Tier {tier_name}: Resolved {count} tokens.")

            return current_results

        # Tier 1: Trigram
        results = apply_tier(
            results, self.trigram_map, ["prev", "before", "next"], "Trigram"
        )

        # Tier 2: Bigram Prev
        results = apply_tier(
            results, self.bigram_prev_map, ["prev", "before"], "Bigram-Prev"
        )

        # Tier 3: Bigram Next
        results = apply_tier(
            results, self.bigram_next_map, ["before", "next"], "Bigram-Next"
        )

        # Tier 4: Unigram
        results = apply_tier(results, self.unigram_map, ["before"], "Unigram")

        remaining = results.isna().sum()
        self.logger.info(
            f"HFBB Inference Complete. Unresolved: {remaining} ({remaining/len(df):.2%})"
        )

        return results
