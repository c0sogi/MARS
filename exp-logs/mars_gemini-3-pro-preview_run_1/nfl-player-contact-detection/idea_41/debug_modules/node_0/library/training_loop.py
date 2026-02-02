import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, matthews_corrcoef
import joblib

from library.config import CACHE_DIR, SEED
from library.utils import (
    setup_logger,
    CacheManager,
    optimize_threshold,
    seed_everything,
)
from library.data_processing import DataProcessor
from library.feature_engineering import FeatureEngineer
from library.mining_strategy import MiningStrategy
from library.model_definitions import LGBMExpert, XGBExpert


class TrainingLoop:
    """
    Orchestrates the KAT-AME training pipeline:
    1. Data Loading & Feature Engineering
    2. Dual-Scout Training & Hard Negative Mining
    3. Expert Model Training on Anchored Dataset
    4. Validation & Threshold Optimization
    """

    def __init__(self, logger=None):
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "training_loop.log"))
        )
        self.cache_manager = CacheManager()
        self.model_dir = os.path.join(CACHE_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def run_training(self, load_cached_data=True, load_cached_models=True):
        """
        Executes the full training pipeline.

        Args:
            load_cached_data (bool): Whether to load features from cache.
            load_cached_models (bool): Whether to load Scout models from cache.
        """
        seed_everything(SEED)
        self.logger.info("Starting Training Loop...")

        # =========================================================================
        # 1. Data Loading & Feature Engineering
        # =========================================================================
        processor = DataProcessor(logger=self.logger)
        engineer = FeatureEngineer(logger=self.logger)

        # --- Load Metadata ---
        df_meta_train = processor.load_metadata(split="train")
        df_meta_val = processor.load_metadata(split="val")

        # --- Load Tracking ---
        df_tracking = processor.load_tracking(split="train")

        # --- Generate Features (Train) ---
        df_features_train = engineer.create_features(
            df_meta_train,
            df_tracking,
            split="train",
            load_cached_data=load_cached_data,
        )

        # --- Generate Features (Val) ---
        df_features_val = engineer.create_features(
            df_meta_val,
            df_tracking,
            split="val",
            load_cached_data=load_cached_data,
        )

        # --- Prepare X and y ---
        # Identify feature columns (drop metadata)
        exclude_cols = [
            "contact_id",
            "contact",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        feature_cols = [c for c in df_features_train.columns if c not in exclude_cols]

        self.logger.info(f"Number of features: {len(feature_cols)}")

        X_train = df_features_train[feature_cols]
        y_train = df_features_train["contact"]

        X_val = df_features_val[feature_cols]
        y_val = df_features_val["contact"]

        # Clean up memory
        del df_features_train, df_features_val, df_meta_train, df_meta_val, df_tracking
        import gc

        gc.collect()

        # =========================================================================
        # 2. Mining Strategy (Scouts -> Hard Negatives -> Expert Dataset)
        # =========================================================================
        miner = MiningStrategy(logger=self.logger)

        # Train Scouts (Balanced)
        scout_lgbm, scout_xgb = miner.train_scouts(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=feature_cols,
            load_cached_models=load_cached_models,
        )

        # Mine Hard Negatives
        hard_neg_indices = miner.mine_hard_negatives(
            scout_lgbm,
            scout_xgb,
            X_train,
            y_train,
            load_cached_indices=load_cached_models,
        )

        # Construct Expert Dataset
        X_expert, y_expert = miner.construct_expert_dataset(
            X_train, y_train, hard_neg_indices
        )

        # =========================================================================
        # 3. Train Expert Models
        # =========================================================================
        self.logger.info("Initializing Expert Models...")

        expert_lgbm = LGBMExpert(logger=self.logger)
        expert_xgb = XGBExpert(logger=self.logger)

        # Train Expert LGBM
        self.logger.info("Training Expert LGBM...")
        expert_lgbm.fit(X_expert, y_expert, X_val, y_val, feature_names=feature_cols)
        expert_lgbm.save(os.path.join(self.model_dir, "expert_lgbm.joblib"))

        # Train Expert XGB
        self.logger.info("Training Expert XGB...")
        expert_xgb.fit(X_expert, y_expert, X_val, y_val, feature_names=feature_cols)
        expert_xgb.save(os.path.join(self.model_dir, "expert_xgb.joblib"))

        # =========================================================================
        # 4. Validation & Threshold Optimization
        # =========================================================================
        self.logger.info("Evaluating Expert Ensemble on Validation Set...")

        # Predict Probabilities
        preds_lgbm = expert_lgbm.predict_proba(X_val)
        preds_xgb = expert_xgb.predict_proba(X_val)

        # Ensemble (Simple Average)
        preds_ensemble = (preds_lgbm + preds_xgb) / 2.0

        # Calculate LogLoss
        val_logloss = log_loss(y_val, preds_ensemble)
        self.logger.info(f"Ensemble Validation LogLoss: {val_logloss}")

        # Optimize Threshold
        best_threshold, best_mcc = optimize_threshold(y_val.values, preds_ensemble)

        self.logger.info(f"Best Threshold: {best_threshold}")
        self.logger.info(f"Best MCC Score: {best_mcc}")

        # Save Threshold
        threshold_path = os.path.join(self.model_dir, "best_threshold.npy")
        np.save(threshold_path, np.array([best_threshold]))
        self.logger.info(f"Saved best threshold to {threshold_path}")

        return best_threshold, best_mcc
