import pandas as pd
import numpy as np
import os
import gc
import joblib
from library.config import (
    SCOUT_TRAIN_RATIO,
    HARD_NEGATIVE_THRESHOLD,
    EXPERT_BUFFER_RATIO,
    SEED,
    IDEA_DIR,
)
from library.utils import setup_logger, CacheManager
from library.model_definitions import LGBMExpert, XGBExpert, EnsemblePredictor


class Trainer:
    """
    Implements the Dual-Scout Diversity Mining Curriculum.
    Phase 1: Train Scouts (LGBM + XGB) on balanced data.
    Phase 2: Mine Hard Negatives (Union of Scout errors).
    Phase 3: Train Expert Tri-Ensemble on Positives + Hard Negatives + Buffer.
    """

    def __init__(self):
        self.logger = setup_logger("trainer")
        self.cache_manager = CacheManager(cache_dir=IDEA_DIR)
        self.scout_lgbm = None
        self.scout_xgb = None
        self.ensemble = None

    def _get_feature_cols(self, df):
        """Identifies feature columns by excluding metadata and targets."""
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "offset",
            "step_actual",
        ]
        # Select numeric columns that are not in the exclude list
        return [
            c
            for c in df.columns
            if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
        ]

    def train_scouts(self, df_train, feature_cols, target_col="contact"):
        """
        Phase 1: Train Scout models on a balanced subset of the data.
        """
        self.logger.info("Phase 1: Training Scouts...")

        # 1. Create Balanced Scout Dataset
        df_pos = df_train[df_train[target_col] == 1]
        df_neg = df_train[df_train[target_col] == 0]

        n_pos = len(df_pos)
        n_neg_sample = int(n_pos * SCOUT_TRAIN_RATIO)

        # Sample negatives
        if len(df_neg) > n_neg_sample:
            df_neg_sample = df_neg.sample(n=n_neg_sample, random_state=SEED)
        else:
            df_neg_sample = df_neg

        df_scout = (
            pd.concat([df_pos, df_neg_sample], axis=0)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

        X_scout = df_scout[feature_cols]
        y_scout = df_scout[target_col]

        self.logger.info(
            f"Scout Training Set: {len(df_scout)} samples (Pos: {len(df_pos)}, Neg: {len(df_neg_sample)})"
        )

        # 2. Train Scout A (LGBM)
        self.logger.info("Training Scout A (LGBM)...")
        self.scout_lgbm = LGBMExpert()
        # We don't use validation set for scouts to keep them diverse and slightly overfit to the balanced subset
        self.scout_lgbm.fit(X_scout, y_scout)

        # 3. Train Scout B (XGB)
        self.logger.info("Training Scout B (XGB)...")
        self.scout_xgb = XGBExpert()
        self.scout_xgb.fit(X_scout, y_scout)

        # Save Scouts
        scout_dir = os.path.join(IDEA_DIR, "models")
        os.makedirs(scout_dir, exist_ok=True)
        self.scout_lgbm.save(os.path.join(scout_dir, "scout_lgbm.joblib"))
        self.scout_xgb.save(os.path.join(scout_dir, "scout_xgb.joblib"))

        return self.scout_lgbm, self.scout_xgb

    def mine_hard_negatives(
        self, df_train, feature_cols, target_col="contact", load_cached_data=True
    ):
        """
        Phase 2: Mine Hard Negatives using the trained Scouts.
        Hard Negatives are negative samples where either scout predicts > threshold.
        """
        # Cache Check
        # We hash the dataframe length and the threshold to ensure validity
        cache_params = {
            "df_len": len(df_train),
            "threshold": HARD_NEGATIVE_THRESHOLD,
            "stage": "hard_negative_indices",
        }

        if load_cached_data:
            indices = self.cache_manager.load("hard_negative_indices", cache_params)
            if indices is not None:
                self.logger.info(
                    f"Loaded {len(indices)} hard negative indices from cache."
                )
                return indices

        self.logger.info("Phase 2: Mining Hard Negatives...")

        if self.scout_lgbm is None or self.scout_xgb is None:
            raise ValueError("Scouts must be trained before mining hard negatives.")

        # Filter to only negatives in the training set
        # We keep the original index to return indices relative to df_train
        neg_mask = df_train[target_col] == 0
        df_neg_all = df_train[neg_mask]

        if len(df_neg_all) == 0:
            self.logger.warning("No negatives found in training set.")
            return np.array([])

        X_neg = df_neg_all[feature_cols]

        # Predict with both scouts
        self.logger.info(f"Scoring {len(X_neg)} negative samples...")
        probs_lgbm = self.scout_lgbm.predict_proba(X_neg)
        probs_xgb = self.scout_xgb.predict_proba(X_neg)

        # Compute Union of Errors (Max Probability)
        probs_max = np.maximum(probs_lgbm, probs_xgb)

        # Identify Hard Negatives
        hard_mask = probs_max > HARD_NEGATIVE_THRESHOLD
        hard_indices = df_neg_all.index[hard_mask].to_numpy()

        self.logger.info(
            f"Mined {len(hard_indices)} Hard Negatives out of {len(df_neg_all)} candidates."
        )

        # Save to Cache
        self.cache_manager.save(hard_indices, "hard_negative_indices", cache_params)

        return hard_indices

    def train_experts(
        self,
        df_train,
        df_val,
        hard_negative_indices,
        feature_cols,
        target_col="contact",
    ):
        """
        Phase 3: Train the Expert Tri-Ensemble on the curriculum dataset.
        Dataset = All Positives + Hard Negatives + Random Buffer.
        """
        self.logger.info("Phase 3: Training Expert Tri-Ensemble...")

        # 1. Construct Expert Dataset
        df_pos = df_train[df_train[target_col] == 1]

        # Hard Negatives
        if len(hard_negative_indices) > 0:
            df_hard_neg = df_train.loc[hard_negative_indices]
        else:
            df_hard_neg = pd.DataFrame(columns=df_train.columns)

        # Buffer Negatives (Random sample from remaining negatives)
        # Exclude hard negatives to avoid duplication
        neg_mask = df_train[target_col] == 0
        all_neg_indices = df_train.index[neg_mask]
        remaining_neg_indices = np.setdiff1d(all_neg_indices, hard_negative_indices)

        n_buffer = int(len(df_pos) * EXPERT_BUFFER_RATIO)

        if len(remaining_neg_indices) > n_buffer:
            buffer_indices = np.random.choice(
                remaining_neg_indices, size=n_buffer, replace=False
            )
            df_buffer_neg = df_train.loc[buffer_indices]
        else:
            df_buffer_neg = df_train.loc[remaining_neg_indices]

        # Combine
        df_expert = (
            pd.concat([df_pos, df_hard_neg, df_buffer_neg], axis=0)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

        self.logger.info(f"Expert Training Set: {len(df_expert)} samples")
        self.logger.info(f"  - Positives: {len(df_pos)}")
        self.logger.info(f"  - Hard Negatives: {len(df_hard_neg)}")
        self.logger.info(f"  - Buffer Negatives: {len(df_buffer_neg)}")

        X_train = df_expert[feature_cols]
        y_train = df_expert[target_col]

        X_val = df_val[feature_cols]
        y_val = df_val[target_col]

        # 2. Initialize and Train Ensemble
        self.ensemble = EnsemblePredictor()
        self.ensemble.fit(X_train, y_train, X_val, y_val)

        return self.ensemble

    def run_curriculum(self, df_train, df_val, load_cached_data=True):
        """
        Executes the full training pipeline.

        Args:
            df_train (pd.DataFrame): Training features.
            df_val (pd.DataFrame): Validation features.
            load_cached_data (bool): Whether to use cached intermediate data (hard negatives).

        Returns:
            EnsemblePredictor: The trained expert ensemble.
        """
        feature_cols = self._get_feature_cols(df_train)
        self.logger.info(f"Identified {len(feature_cols)} feature columns.")

        # Phase 1: Scouts
        # Check if scouts exist on disk to skip training if needed?
        # For this implementation, we retrain scouts to ensure consistency with current data,
        # unless we want to implement model caching. The prompt implies implementing the logic.
        # We will train scouts.
        self.train_scouts(df_train, feature_cols)

        # Phase 2: Mining
        hard_indices = self.mine_hard_negatives(
            df_train, feature_cols, load_cached_data=load_cached_data
        )

        # Phase 3: Experts
        self.train_experts(df_train, df_val, hard_indices, feature_cols)

        self.logger.info("Curriculum Training Complete.")
        return self.ensemble
