import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import setup_logger, seed_everything, generate_content_hash

# Initialize logger
logger = setup_logger("sampler")


class DataSampler:
    """
    Handles dynamic dataset construction for the Iterative Hard-Negative Mining strategy.
    Implements caching for deterministic dataset creation steps.
    """

    def __init__(self):
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR

        # Hyperparameters from Config
        self.scout_neg_ratio = Config.SCOUT_NEG_RATIO
        self.expert_neg_ratio = Config.EXPERT_RANDOM_NEG_RATIO
        self.hard_neg_threshold = Config.HARD_NEGATIVE_THRESHOLD

        # Ensure reproducibility
        seed_everything(self.seed)

    def _get_cache_path(self, name, params):
        """Generates a unique cache filename based on parameters."""
        hash_str = generate_content_hash(params)
        filename = f"dataset_{name}_{hash_str}.parquet"
        return os.path.join(self.working_dir, filename)

    def create_scout_dataset(self, df, target_col="contact", load_cached_data=True):
        """
        Creates the training set for the Scout model (Positives + Random Negatives).

        Args:
            df (pd.DataFrame): The full training features DataFrame.
            target_col (str): The name of the target column.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The balanced (or specifically imbalanced) Scout dataset.
        """
        # Define cache parameters
        params = {
            "type": "scout_dataset",
            "n_total": len(df),
            "neg_ratio": self.scout_neg_ratio,
            "seed": self.seed,
        }
        cache_path = self._get_cache_path("scout", params)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached Scout dataset from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recreating dataset...")

        logger.info("Creating Scout dataset...")

        # 2. Filter Data
        pos_mask = df[target_col] == 1
        neg_mask = df[target_col] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        # 3. Sample Negatives
        n_pos = len(df_pos)
        n_neg_target = int(n_pos * self.scout_neg_ratio)

        if len(df_neg) > n_neg_target:
            df_neg_sampled = df_neg.sample(n=n_neg_target, random_state=self.seed)
        else:
            df_neg_sampled = df_neg

        # 4. Combine and Shuffle
        df_scout = pd.concat([df_pos, df_neg_sampled], axis=0)
        df_scout = df_scout.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        logger.info(
            f"Scout Dataset Created: {len(df_scout)} rows (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
        )

        # 5. Save Cache
        try:
            df_scout.to_parquet(cache_path, index=False)
            logger.info(f"Saved Scout dataset to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return df_scout

    def mine_hard_negatives(self, model, df, feature_cols, target_col="contact"):
        """
        Uses the trained Scout model to identify Hard Negatives from the full dataset.
        Hard Negatives are negative samples where predicted probability > threshold.

        Args:
            model: Trained model instance (must have predict method).
            df (pd.DataFrame): Full training DataFrame.
            feature_cols (list): List of feature column names used for prediction.
            target_col (str): Name of target column.

        Returns:
            pd.DataFrame: Subset of df containing hard negatives.
        """
        logger.info("Mining Hard Negatives from full dataset...")

        # Filter for negatives only
        # We keep the original index to allow exclusion later
        df_neg = df[df[target_col] == 0].copy()

        if df_neg.empty:
            logger.warning("No negatives found in dataset.")
            return df_neg

        # Run Inference
        # Note: We assume the model handles the DataFrame input correctly (via wrapper)
        X_neg = df_neg[feature_cols]
        preds = model.predict(X_neg)

        # Filter based on threshold
        hard_mask = preds > self.hard_neg_threshold
        df_hard = df_neg[hard_mask]

        logger.info(
            f"Mining Complete. Found {len(df_hard)} Hard Negatives out of {len(df_neg)} evaluated."
        )
        logger.info(f"Hard Negative Rate: {len(df_hard)/len(df_neg):.10f}")

        return df_hard

    def create_expert_dataset(
        self, df, hard_negatives_df, target_col="contact", load_cached_data=True
    ):
        """
        Creates the training set for the Expert model.
        Composition: All Positives + Mined Hard Negatives + Buffer of Random Negatives.

        Args:
            df (pd.DataFrame): The full training features DataFrame.
            hard_negatives_df (pd.DataFrame): The DataFrame of mined hard negatives.
            target_col (str): The name of the target column.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The Expert dataset.
        """
        # Define cache parameters
        # We use the count of hard negatives as a proxy for the mining result signature
        params = {
            "type": "expert_dataset",
            "n_total": len(df),
            "n_hard_negs": len(hard_negatives_df),
            "random_ratio": self.expert_neg_ratio,
            "seed": self.seed,
        }
        cache_path = self._get_cache_path("expert", params)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached Expert dataset from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recreating dataset...")

        logger.info("Creating Expert dataset...")

        # 2. Get Positives
        df_pos = df[df[target_col] == 1]

        # 3. Get Hard Negatives
        # We use the provided dataframe.
        df_hard = hard_negatives_df

        # 4. Get Random Buffer Negatives
        # We want to sample from negatives that are NOT in the hard negative set
        # to provide a broader view of the negative class.

        # Identify all negatives
        df_all_neg = df[df[target_col] == 0]

        # Exclude hard negatives using index
        # This assumes hard_negatives_df preserves the index from df (which mine_hard_negatives does)
        hard_indices = df_hard.index

        # Drop hard negatives from the pool
        df_neg_pool = df_all_neg.drop(index=hard_indices, errors="ignore")

        # Sample random buffer
        n_pos = len(df_pos)
        n_random = int(n_pos * self.expert_neg_ratio)

        if len(df_neg_pool) > n_random:
            df_random = df_neg_pool.sample(n=n_random, random_state=self.seed)
        else:
            df_random = df_neg_pool

        # 5. Combine
        df_expert = pd.concat([df_pos, df_hard, df_random], axis=0)

        # Ensure no index duplicates (just in case)
        df_expert = df_expert[~df_expert.index.duplicated(keep="first")]

        # Shuffle
        df_expert = df_expert.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        logger.info(f"Expert Dataset Created: {len(df_expert)} rows")
        logger.info(f"   - Positives: {len(df_pos)}")
        logger.info(f"   - Hard Negatives: {len(df_hard)}")
        logger.info(f"   - Random Negatives: {len(df_random)}")

        # 6. Save Cache
        try:
            df_expert.to_parquet(cache_path, index=False)
            logger.info(f"Saved Expert dataset to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return df_expert
