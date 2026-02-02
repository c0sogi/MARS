import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    setup_logger,
    save_to_parquet,
    load_from_parquet,
    get_experiment_hash,
    seed_everything,
)
from library.model_factory import ModelFactory


class ScoutMiner:
    """
    Implements Phase 1 and 2 of the VRC-ME pipeline:
    1. Train a lightweight 'Scout' model on a balanced subset of Tier 1 features.
    2. Mine 'Hard Negatives' from the full dataset to construct the Expert training set.
    """

    def __init__(self):
        self.logger = setup_logger(name="ScoutMiner")
        self.features = Config.TIER1_FEATURES
        self.target = "contact"
        seed_everything(Config.SEED)

    def _prepare_balanced_data(self, df: pd.DataFrame, neg_ratio: int = 10):
        """
        Creates a balanced training subset: All Positives + (neg_ratio * Positives) Random Negatives.
        """
        self.logger.info("Preparing balanced subset for Scout training...")

        pos_mask = df[self.target] == 1
        neg_mask = df[self.target] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg_sample = min(len(df_neg), n_pos * neg_ratio)

        self.logger.info(
            f"Positives: {n_pos}, Sampling {n_neg_sample} Negatives (Ratio 1:{neg_ratio})"
        )

        df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=Config.SEED)

        df_balanced = (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        return df_balanced

    def _train_scout(self, df_train_balanced: pd.DataFrame, df_val: pd.DataFrame):
        """
        Trains the Scout LightGBM model.
        """
        self.logger.info("Initializing Scout Model (LightGBM)...")
        model = ModelFactory.create_model(stage="scout", model_type="lgbm")

        # Prepare Feature Matrices
        # Ensure we only use Tier 1 features present in the dataframe
        valid_features = [f for f in self.features if f in df_train_balanced.columns]

        X_train = df_train_balanced[valid_features]
        y_train = df_train_balanced[self.target]

        X_val = df_val[valid_features]
        y_val = df_val[self.target]

        self.logger.info(
            f"Training Scout on {len(X_train)} samples with {len(valid_features)} features..."
        )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        return model, valid_features

    def _mine_indices(
        self, model, df_full: pd.DataFrame, feature_cols: list
    ) -> np.ndarray:
        """
        Runs inference on the full dataset and selects:
        1. All True Positives (Contact = 1)
        2. Hard Negatives (Contact = 0 AND Prob > Threshold)
        """
        self.logger.info(
            f"Running Scout inference on full training set ({len(df_full)} rows)..."
        )

        # Predict probabilities
        # Note: predict_proba returns [prob_0, prob_1]
        probs = model.predict_proba(df_full[feature_cols])[:, 1]

        # Identify masks
        # 1. True Positives
        mask_pos = df_full[self.target] == 1

        # 2. Hard Negatives
        # Predicted probability > Threshold AND Actual is 0
        mask_hard_neg = (probs > Config.MINING_THRESHOLD) & (df_full[self.target] == 0)

        # Combine
        final_mask = mask_pos | mask_hard_neg

        # Extract indices (assuming df_full index is aligned with original or we return boolean mask/indices relative to df_full)
        # We will return the indices of the dataframe rows that satisfy the condition.
        selected_indices = df_full.index[final_mask].to_numpy()

        n_pos = mask_pos.sum()
        n_hard = mask_hard_neg.sum()
        total_selected = len(selected_indices)

        self.logger.info(f"Mining Complete.")
        self.logger.info(f"True Positives: {n_pos}")
        self.logger.info(
            f"Hard Negatives Found (Prob > {Config.MINING_THRESHOLD}): {n_hard}"
        )
        self.logger.info(
            f"Total Expert Dataset Size: {total_selected} ({total_selected/len(df_full):.2%} of full data)"
        )

        return selected_indices

    def execute(
        self,
        df_train_tier1: pd.DataFrame,
        df_val_tier1: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> np.ndarray:
        """
        Main execution method for the Mining module.

        Args:
            df_train_tier1: Full training dataframe with Tier 1 features.
            df_val_tier1: Validation dataframe with Tier 1 features.
            load_cached_data: Whether to load mined indices from cache.

        Returns:
            np.ndarray: Array of indices from df_train_tier1 to be used for Expert training.
        """
        # 1. Generate Cache Key
        # Hash based on data shape, threshold, and random seed
        params = {
            "train_shape": df_train_tier1.shape,
            "val_shape": df_val_tier1.shape,
            "mining_threshold": Config.MINING_THRESHOLD,
            "seed": Config.SEED,
            "neg_ratio": 10,
        }
        cache_hash = get_experiment_hash(params)
        cache_filename = f"mined_indices_{cache_hash}.parquet"

        # 2. Check Cache
        if load_cached_data:
            cached_df = load_from_parquet(cache_filename)
            if cached_df is not None:
                self.logger.info(f"Loaded mined indices from cache: {cache_filename}")
                return cached_df["index"].values

        self.logger.info("Cache miss or force reload. Starting Mining Pipeline...")

        # 3. Prepare Data
        df_balanced = self._prepare_balanced_data(df_train_tier1, neg_ratio=10)

        # 4. Train Scout
        model, valid_features = self._train_scout(df_balanced, df_val_tier1)

        # 5. Mine Hard Negatives
        mined_indices = self._mine_indices(model, df_train_tier1, valid_features)

        # 6. Save to Cache
        # Save as a simple dataframe with one column
        df_indices = pd.DataFrame({"index": mined_indices})
        save_to_parquet(df_indices, cache_filename)
        self.logger.info(f"Saved mined indices to cache: {cache_filename}")

        return mined_indices
