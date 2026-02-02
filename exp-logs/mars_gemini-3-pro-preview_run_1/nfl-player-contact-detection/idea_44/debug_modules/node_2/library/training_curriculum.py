import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, log_loss
from typing import Tuple, List, Dict, Any

from library.config import Config
from library.utils import setup_logging, seed_everything, CacheManager
from library.data_processing import DataLoader
from library.feature_engineering import KinematicFeatureEngine
from library.models import LGBMWrapper, XGBWrapper


class CurriculumManager:
    """
    Orchestrates the KARP-AM (Kinematically-Aligned Relative-Physics with Anchored Mining)
    training strategy.

    Phases:
    1. Dual-Scout Training: Train initial models on balanced data.
    2. Percentile-Based Mining: Identify Hard Negatives from the full dataset.
    3. Anchored Expert Training: Train final models on Positives + Hard Negatives + Anchors.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.cache = CacheManager()
        self.data_loader = DataLoader()
        self.feature_engine = KinematicFeatureEngine()

        # Define model output directory
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, load_cached_data: bool = True):
        """
        Executes the full training curriculum.
        """
        seed_everything(Config.SEED)
        self.logger.info("Starting KARP-AM Training Curriculum...")

        # ---------------------------------------------------------------------
        # 1. Data Loading & Feature Engineering
        # ---------------------------------------------------------------------
        # Load and process Train
        meta_train, track_train = self.data_loader.load_data(
            "train", load_cached_data=load_cached_data
        )

        # Debugging: Sample data if configured
        if Config.DEBUG:
            self.logger.info(
                f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from train metadata."
            )
            meta_train = meta_train.sample(
                n=min(len(meta_train), Config.DEBUG_SAMPLE_SIZE),
                random_state=Config.SEED,
            )

        df_train = self.feature_engine.process_data(
            meta_train,
            track_train,
            dataset_key="train",
            load_cached_data=load_cached_data,
        )

        # Load and process Val
        meta_val, track_val = self.data_loader.load_data(
            "val", load_cached_data=load_cached_data
        )

        if Config.DEBUG:
            meta_val = meta_val.sample(
                n=min(len(meta_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        df_val = self.feature_engine.process_data(
            meta_val, track_val, dataset_key="val", load_cached_data=load_cached_data
        )

        # Clean up raw tracking data to save memory
        del meta_train, track_train, meta_val, track_val
        gc.collect()

        # ---------------------------------------------------------------------
        # 2. Mining Phase (Scouts -> Hard Negatives)
        # ---------------------------------------------------------------------
        # Check if hard negatives are already cached
        hn_cache_key = f"hard_negative_indices_{Config.EXP_NAME}"
        hard_negative_indices = None

        if load_cached_data:
            cached_hn = self.cache.load(hn_cache_key + ".npy")
            if cached_hn is not None:
                self.logger.info("Loaded cached Hard Negative indices.")
                hard_negative_indices = cached_hn

        if hard_negative_indices is None:
            self.logger.info("Phase 1: Training Scouts for Hard Negative Mining...")
            scout_lgbm, scout_xgb = self._train_scouts(df_train)

            self.logger.info("Phase 2: Mining Hard Negatives (Percentile-Based)...")
            hard_negative_indices = self._mine_hard_negatives(
                scout_lgbm, scout_xgb, df_train
            )

            # Cache the indices
            self.cache.save(hard_negative_indices, hn_cache_key + ".npy")

            # Clean up scouts
            del scout_lgbm, scout_xgb
            gc.collect()

        # ---------------------------------------------------------------------
        # 3. Expert Training Phase
        # ---------------------------------------------------------------------
        self.logger.info("Phase 3: Training Anchored Experts...")
        expert_lgbm, expert_xgb = self._train_experts(
            df_train, df_val, hard_negative_indices
        )

        self.logger.info("Training Curriculum Completed.")
        return expert_lgbm, expert_xgb

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separates features and target. Drops metadata columns.
        """
        # Metadata columns to exclude from training
        drop_cols = [
            "contact_id",
            "game_play",
            "step",
            "contact",
            "datetime",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]

        feature_cols = [c for c in df.columns if c not in drop_cols]

        X = df[feature_cols]
        y = df["contact"]

        return X, y

    def _train_scouts(self, df_train: pd.DataFrame) -> Tuple[LGBMWrapper, XGBWrapper]:
        """
        Trains Scout models on a balanced subset (1:1 Positive:RandomNegative).
        """
        # Separate Positives and Negatives
        pos_mask = df_train["contact"] == 1
        neg_mask = df_train["contact"] == 0

        df_pos = df_train[pos_mask]
        df_neg = df_train[neg_mask]

        n_pos = len(df_pos)

        # Sample Negatives (1:1 ratio)
        df_neg_sample = df_neg.sample(n=n_pos, random_state=Config.SEED)

        # Create Scout Dataset
        df_scout = (
            pd.concat([df_pos, df_neg_sample])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        X_scout, y_scout = self._prepare_xy(df_scout)

        self.logger.info(
            f"Scout Dataset Size: {len(df_scout)} (Pos: {n_pos}, Neg: {len(df_neg_sample)})"
        )

        # Train Scout A (LGBM)
        scout_lgbm = LGBMWrapper(name="scout_lgbm")
        scout_lgbm.fit(X_scout, y_scout)

        # Train Scout B (XGB)
        scout_xgb = XGBWrapper(name="scout_xgb")
        scout_xgb.fit(X_scout, y_scout)

        return scout_lgbm, scout_xgb

    def _mine_hard_negatives(
        self, scout_lgbm: LGBMWrapper, scout_xgb: XGBWrapper, df_train: pd.DataFrame
    ) -> np.ndarray:
        """
        Uses Scouts to predict on ALL negatives in df_train.
        Selects Top-K False Positives (Hard Negatives).
        """
        # Filter only negatives for mining
        neg_indices = df_train[df_train["contact"] == 0].index
        X_neg, _ = self._prepare_xy(df_train.loc[neg_indices])

        self.logger.info(f"Predicting on {len(X_neg)} negative samples for mining...")

        # Get probabilities
        p_lgbm = scout_lgbm.predict_proba(X_neg)
        p_xgb = scout_xgb.predict_proba(X_neg)

        # Ensemble average
        p_avg = (p_lgbm + p_xgb) / 2.0

        # Determine K
        n_pos = df_train["contact"].sum()
        k = int(n_pos * Config.HARD_NEGATIVE_RATIO)

        self.logger.info(f"Selecting Top-{k} Hard Negatives...")

        # Get indices of top K probabilities
        # argsort is ascending, so take last K and reverse
        top_k_local_indices = np.argsort(p_avg)[-k:][::-1]

        # Map back to original dataframe indices
        hard_negative_indices = neg_indices[top_k_local_indices].values

        return hard_negative_indices

    def _train_experts(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, hard_neg_indices: np.ndarray
    ) -> Tuple[LGBMWrapper, XGBWrapper]:
        """
        Trains Expert models on:
        1. All Positives
        2. Mined Hard Negatives
        3. Random Anchors (Easy Negatives)
        """
        # 1. Positives
        pos_mask = df_train["contact"] == 1
        df_pos = df_train[pos_mask]
        n_pos = len(df_pos)

        # 2. Hard Negatives
        df_hard = df_train.loc[hard_neg_indices]

        # 3. Random Anchors
        # Exclude hard negatives from the pool of potential anchors
        # Note: df_train indices are unique per row
        all_neg_indices = df_train[df_train["contact"] == 0].index
        potential_anchor_indices = np.setdiff1d(all_neg_indices, hard_neg_indices)

        n_anchors = int(n_pos * Config.ANCHOR_RATIO)
        # Ensure we don't sample more than available
        n_anchors = min(n_anchors, len(potential_anchor_indices))

        anchor_indices = np.random.choice(
            potential_anchor_indices, size=n_anchors, replace=False
        )
        df_anchors = df_train.loc[anchor_indices]

        # Combine
        df_expert = (
            pd.concat([df_pos, df_hard, df_anchors])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        self.logger.info(f"Expert Dataset Composition:")
        self.logger.info(f"  Positives: {len(df_pos)}")
        self.logger.info(f"  Hard Negatives: {len(df_hard)}")
        self.logger.info(f"  Anchors: {len(df_anchors)}")
        self.logger.info(f"  Total: {len(df_expert)}")

        X_train, y_train = self._prepare_xy(df_expert)
        X_val, y_val = self._prepare_xy(df_val)

        # Train Expert A (LGBM)
        expert_lgbm = LGBMWrapper(name="expert_lgbm")
        expert_lgbm.fit(X_train, y_train, X_val, y_val)
        self._evaluate_model(expert_lgbm, X_val, y_val)
        expert_lgbm.save(self.model_dir)

        # Train Expert B (XGB)
        expert_xgb = XGBWrapper(name="expert_xgb")
        expert_xgb.fit(X_train, y_train, X_val, y_val)
        self._evaluate_model(expert_xgb, X_val, y_val)
        expert_xgb.save(self.model_dir)

        return expert_lgbm, expert_xgb

    def _evaluate_model(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Evaluates model on validation set and prints MCC.
        """
        probs = model.predict_proba(X_val)

        # Find best threshold for MCC
        thresholds = np.linspace(0.1, 0.9, 81)
        best_mcc = -1
        best_thresh = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = matthews_corrcoef(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        self.logger.info(
            f"[{model.name}] Validation MCC: {best_mcc} (Threshold: {best_thresh})"
        )
