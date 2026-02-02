import pandas as pd
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle

from library.config import (
    SCOUT_LGBM_PARAMS,
    EXPERT_LGBM_PARAMS,
    EXPERT_XGB_PARAMS,
    MINING_THRESHOLD,
    SEED,
)
import library.config as config
from library.utils import setup_logging, CacheManager, seed_everything
from library.data_factory import DataFactory
from library.model_zoo import LGBMWrapper, EnsembleModel


class MiningTrainer:
    def __init__(self):
        """
        Initializes the trainer with data factories, cache manager, and logging.
        """
        self.logger = setup_logging()
        self.cache = CacheManager(
            cache_dir=os.path.join(config.WORKING_DIR, "mining_cache")
        )

        # Data Factories
        self.train_data_factory = DataFactory(mode="train")
        self.val_data_factory = DataFactory(mode="val")

        # Columns to exclude from feature matrix X
        self.ignore_cols = [
            "contact",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "p2_int",
            "step_join",
            "step_temp",
        ]

    def _get_features_and_target(self, df):
        """
        Helper to separate feature matrix X and target vector y from a DataFrame.
        """
        y = df["contact"].values
        # Drop metadata columns to retain only features (including is_ground)
        drop_cols = [c for c in self.ignore_cols if c in df.columns]
        X = df.drop(columns=drop_cols)
        return X, y

    def train_scout(self, df_train):
        """
        Phase 1: Train Scout Model (LightGBM) on a balanced subset of the gated training data.
        Objective: Learn a coarse decision boundary to identify hard negatives.
        """
        self.logger.info("Phase 1: Training Scout Model...")

        # 1. Balance Data (1:1 Ratio)
        pos_mask = df_train["contact"] == 1
        neg_mask = df_train["contact"] == 0

        df_pos = df_train[pos_mask]
        df_neg = df_train[neg_mask]

        # Sample negatives equal to the number of positives
        n_pos = len(df_pos)
        if len(df_neg) > n_pos:
            df_neg_sampled = df_neg.sample(n=n_pos, random_state=SEED)
        else:
            df_neg_sampled = df_neg

        df_balanced = pd.concat([df_pos, df_neg_sampled])
        df_balanced = shuffle(df_balanced, random_state=SEED).reset_index(drop=True)

        X_scout, y_scout = self._get_features_and_target(df_balanced)

        # 2. Create Internal Validation Split for Early Stopping (80/20)
        split_idx = int(len(X_scout) * 0.8)
        X_tr, y_tr = X_scout.iloc[:split_idx], y_scout[:split_idx]
        X_val, y_val = X_scout.iloc[split_idx:], y_scout[split_idx:]

        # 3. Train Model
        scout_model = LGBMWrapper(SCOUT_LGBM_PARAMS, model_name="scout_lgbm")
        scout_model.fit(X_tr, y_tr, X_val, y_val)
        scout_model.save()

        return scout_model

    def mine_hard_negatives(self, scout_model, df_train, load_cached=True):
        """
        Phase 2: Mine Hard Negatives.
        Runs inference on the FULL gated training set to find non-contacts with high predicted probability.
        """
        self.logger.info("Phase 2: Mining Hard Negatives...")

        def _compute_mining():
            # Prepare full training set for inference
            X_full, y_full = self._get_features_and_target(df_train)

            # Predict probabilities
            preds = scout_model.predict(X_full)

            # Identify Hard Negatives: Ground Truth = 0 AND Prediction > Threshold
            hard_neg_mask = (y_full == 0) & (preds > MINING_THRESHOLD)
            hard_neg_indices = df_train.index[hard_neg_mask].to_numpy()

            self.logger.info(
                f"Mining Complete. Found {len(hard_neg_indices)} hard negatives."
            )
            return hard_neg_indices

        return self.cache.execute_with_cache(
            "hard_negative_indices.npy", _compute_mining, load_cached_data=load_cached
        )

    def train_expert(self, df_train, hard_neg_indices, df_val):
        """
        Phase 3: Train Expert Ensemble (LGBM + XGB) on High-Fidelity Dataset.
        Dataset Composition: All Positives + All Mined Hard Negatives + Random Buffer.
        """
        self.logger.info("Phase 3: Training Expert Ensemble...")

        # 1. Construct Expert Dataset
        # All Positives
        df_pos = df_train[df_train["contact"] == 1]

        # All Mined Hard Negatives
        df_hard_neg = df_train.loc[hard_neg_indices]

        # Random Buffer (Sample from all negatives to prevent catastrophic forgetting of easy cases)
        # Size equal to positives
        df_neg_all = df_train[df_train["contact"] == 0]
        n_buffer = len(df_pos)

        # Ensure we don't sample more than available
        if len(df_neg_all) > n_buffer:
            df_buffer = df_neg_all.sample(n=n_buffer, random_state=SEED)
        else:
            df_buffer = df_neg_all

        # Combine and Shuffle
        df_expert = pd.concat([df_pos, df_hard_neg, df_buffer])
        df_expert = shuffle(df_expert, random_state=SEED).reset_index(drop=True)

        self.logger.info(f"Expert Training Set Size: {len(df_expert)}")
        self.logger.info(
            f"Composition: {len(df_pos)} Pos, {len(df_hard_neg)} Hard Neg, {len(df_buffer)} Buffer."
        )

        # Prepare Data
        X_train, y_train = self._get_features_and_target(df_expert)
        X_val_feat, y_val_feat = self._get_features_and_target(df_val)

        # 2. Train Ensemble
        ensemble = EnsembleModel(EXPERT_LGBM_PARAMS, EXPERT_XGB_PARAMS)
        ensemble.fit(X_train, y_train, X_val_feat, y_val_feat)

        return ensemble

    def optimize_threshold(self, ensemble, df_val):
        """
        Optimizes the classification threshold to maximize Matthews Correlation Coefficient (MCC)
        on the validation set.
        """
        self.logger.info("Optimizing Decision Threshold...")

        X_val, y_val = self._get_features_and_target(df_val)
        probs = ensemble.predict(X_val)

        best_threshold = 0.5
        best_mcc = -1.0

        # Search space: 0.10 to 0.90
        thresholds = np.arange(0.1, 0.9, 0.01)

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_val, preds)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        self.logger.info(f"Best Threshold: {best_threshold:.4f}")
        self.logger.info(f"Best Validation MCC: {best_mcc:.10f}")

        return best_threshold

    def run(self):
        """
        Orchestrates the full training curriculum.
        """
        seed_everything(SEED)

        # 1. Load Data
        # Train data is Geometrically Gated (Stage 0)
        df_train = self.train_data_factory.get_train_dataset()
        # Val data is Full (No Gating) for accurate evaluation
        df_val = self.val_data_factory.get_val_dataset()

        # 2. Phase 1: Train Scout
        scout_model = self.train_scout(df_train)

        # 3. Phase 2: Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(scout_model, df_train)

        # 4. Phase 3: Train Expert Ensemble
        expert_ensemble = self.train_expert(df_train, hard_neg_indices, df_val)

        # 5. Optimize Threshold
        best_threshold = self.optimize_threshold(expert_ensemble, df_val)

        # Save Threshold for Inference
        self.cache.save(np.array([best_threshold]), "best_threshold.npy")

        return expert_ensemble, best_threshold
