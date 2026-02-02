import os
import numpy as np
import pandas as pd
from library.config import CACHE_DIR, SEED, ANCHOR_RATIO
from library.utils import setup_logger, CacheManager, seed_everything
from library.model_definitions import LGBMExpert, XGBExpert


class MiningStrategy:
    """
    Implements the Dual-Scout Anchored Mining Curriculum.
    1. Trains Scout models on balanced data.
    2. Mines Hard Negatives (Union of Scout predictions > 0.05).
    3. Constructs Expert Dataset (Positives + Hard Negatives + Anchor Negatives).
    """

    def __init__(self, logger=None):
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "mining_strategy.log"))
        )
        self.cache_manager = CacheManager()
        self.model_dir = os.path.join(CACHE_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train_scouts(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=None,
        load_cached_models=True,
    ):
        """
        Trains Scout A (LGBM) and Scout B (XGB) on a balanced subset of the training data.

        Args:
            X_train (pd.DataFrame): Full training features.
            y_train (pd.Series): Full training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            feature_names (list): List of feature names.
            load_cached_models (bool): Whether to load pre-trained scouts from disk.

        Returns:
            tuple: (scout_lgbm, scout_xgb) trained model instances.
        """
        seed_everything(SEED)

        lgbm_path = os.path.join(self.model_dir, "scout_lgbm.joblib")
        xgb_path = os.path.join(self.model_dir, "scout_xgb.joblib")

        scout_lgbm = LGBMExpert(logger=self.logger)
        scout_xgb = XGBExpert(logger=self.logger)

        # 1. Try Loading from Cache
        if (
            load_cached_models
            and os.path.exists(lgbm_path)
            and os.path.exists(xgb_path)
        ):
            self.logger.info("Loading Scout models from cache...")
            scout_lgbm.load(lgbm_path)
            scout_xgb.load(xgb_path)
            return scout_lgbm, scout_xgb

        self.logger.info("Training Scouts (Cache miss or force retrain)...")

        # 2. Create Balanced Dataset for Scouts
        # Scouts need to learn basic decision boundaries, so we balance 1:1
        pos_mask = y_train == 1
        neg_mask = y_train == 0

        X_pos = X_train[pos_mask]
        y_pos = y_train[pos_mask]

        X_neg = X_train[neg_mask]
        y_neg = y_train[neg_mask]

        # Sample negatives to match positives
        n_pos = len(X_pos)
        if len(X_neg) > n_pos:
            indices = np.random.choice(len(X_neg), size=n_pos, replace=False)
            X_neg_balanced = X_neg.iloc[indices]
            y_neg_balanced = y_neg.iloc[indices]
        else:
            X_neg_balanced = X_neg
            y_neg_balanced = y_neg

        X_scout = pd.concat([X_pos, X_neg_balanced])
        y_scout = pd.concat([y_pos, y_neg_balanced])

        # Shuffle
        shuffle_idx = np.random.permutation(len(X_scout))
        X_scout = X_scout.iloc[shuffle_idx]
        y_scout = y_scout.iloc[shuffle_idx]

        self.logger.info(f"Scout Training Data Shape: {X_scout.shape}")

        # 3. Train Scout A (LGBM)
        self.logger.info("Training Scout A (LGBM)...")
        scout_lgbm.fit(X_scout, y_scout, X_val, y_val, feature_names)
        scout_lgbm.save(lgbm_path)

        # 4. Train Scout B (XGB)
        self.logger.info("Training Scout B (XGB)...")
        scout_xgb.fit(X_scout, y_scout, X_val, y_val, feature_names)
        scout_xgb.save(xgb_path)

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(
        self, scout_lgbm, scout_xgb, X_train, y_train, load_cached_indices=True
    ):
        """
        Runs Scouts on the entire set of negatives in X_train to identify Hard Negatives.
        Hard Negative Definition: Any negative sample where P(Contact) > 0.05 by EITHER Scout.

        Args:
            scout_lgbm: Trained LGBM Scout.
            scout_xgb: Trained XGB Scout.
            X_train (pd.DataFrame): Full training features.
            y_train (pd.Series): Full training labels.
            load_cached_indices (bool): Whether to load indices from cache.

        Returns:
            np.ndarray: Array of indices (relative to X_train) representing Hard Negatives.
        """
        cache_file = "hard_negative_indices.npy"

        if load_cached_indices and self.cache_manager.exists(cache_file):
            self.logger.info(f"Loading hard negative indices from cache: {cache_file}")
            return self.cache_manager.load_numpy(cache_file)

        self.logger.info("Mining Hard Negatives (Inference on full negative pool)...")

        # Filter only negatives
        neg_mask = y_train == 0
        X_neg_pool = X_train[neg_mask]

        # Keep track of original indices to return valid references
        original_indices = X_neg_pool.index.values

        if len(X_neg_pool) == 0:
            self.logger.warning(
                "No negatives found in training set. Returning empty array."
            )
            return np.array([])

        # Predict with Scout A
        preds_lgbm = scout_lgbm.predict_proba(X_neg_pool)

        # Predict with Scout B
        preds_xgb = scout_xgb.predict_proba(X_neg_pool)

        # Apply Union Logic: P > 0.05 in either model
        mining_threshold = 0.05
        hard_mask = (preds_lgbm > mining_threshold) | (preds_xgb > mining_threshold)

        # Extract original indices of hard negatives
        hard_negative_indices = original_indices[hard_mask]

        count = len(hard_negative_indices)
        total = len(X_neg_pool)
        self.logger.info(
            f"Mined {count} Hard Negatives out of {total} ({count/total:.2%})"
        )

        # Cache results
        self.cache_manager.save_numpy(hard_negative_indices, cache_file)

        return hard_negative_indices

    def construct_expert_dataset(self, X_train, y_train, hard_negative_indices):
        """
        Constructs the final training set for the Expert models.
        Composition:
        1. All Positives.
        2. All Mined Hard Negatives.
        3. Random Sample of Easy Negatives (Anchors) at 1:1 ratio with Positives.

        Args:
            X_train (pd.DataFrame): Full training features.
            y_train (pd.Series): Full training labels.
            hard_negative_indices (np.ndarray): Indices of hard negatives in X_train.

        Returns:
            tuple: (X_expert, y_expert)
        """
        seed_everything(SEED)
        self.logger.info("Constructing Expert Dataset...")

        # 1. Identify Positives
        pos_mask = y_train == 1
        X_pos = X_train[pos_mask]
        y_pos = y_train[pos_mask]
        n_pos = len(X_pos)

        # 2. Retrieve Hard Negatives
        # Ensure indices are valid (intersection with current X_train index)
        valid_hard_indices = np.intersect1d(hard_negative_indices, X_train.index.values)
        X_hard = X_train.loc[valid_hard_indices]
        y_hard = y_train.loc[valid_hard_indices]
        n_hard = len(X_hard)

        # 3. Sample Anchors (Easy Negatives)
        # Candidates are negatives that are NOT in hard_negative_indices
        # We can use set difference on indices
        neg_mask = y_train == 0
        all_neg_indices = X_train[neg_mask].index.values

        # Easy negatives = All Negatives - Hard Negatives
        easy_neg_indices = np.setdiff1d(all_neg_indices, valid_hard_indices)

        # Calculate number of anchors needed
        # Ratio defined in config (usually 1.0)
        n_anchors = int(n_pos * ANCHOR_RATIO)

        if len(easy_neg_indices) > n_anchors:
            anchor_indices = np.random.choice(
                easy_neg_indices, size=n_anchors, replace=False
            )
        else:
            anchor_indices = easy_neg_indices
            self.logger.warning(
                f"Not enough easy negatives for requested anchor ratio. Using all {len(easy_neg_indices)} available."
            )

        X_anchors = X_train.loc[anchor_indices]
        y_anchors = y_train.loc[anchor_indices]

        # 4. Combine
        X_expert = pd.concat([X_pos, X_hard, X_anchors])
        y_expert = pd.concat([y_pos, y_hard, y_anchors])

        # Shuffle
        shuffle_idx = np.random.permutation(len(X_expert))
        X_expert = X_expert.iloc[shuffle_idx]
        y_expert = y_expert.iloc[shuffle_idx]

        self.logger.info(f"Expert Dataset Stats:")
        self.logger.info(f"  Positives: {n_pos}")
        self.logger.info(f"  Hard Negatives: {n_hard}")
        self.logger.info(f"  Anchors: {len(X_anchors)}")
        self.logger.info(f"  Total: {len(X_expert)}")

        return X_expert, y_expert
