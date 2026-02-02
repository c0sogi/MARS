import numpy as np
import pandas as pd
import os
import gc
from sklearn.metrics import matthews_corrcoef
from library.config import KADM_CONFIG
from library.utils import setup_logger, seed_everything, process_with_cache
from library.model_zoo import LGBMWrapper, XGBWrapper, DualModelEnsemble
from library.data_loader import DataLoader

# Setup logger
logger = setup_logger(name="training_curriculum")


class ScoutTrainer:
    """
    Trains lightweight 'Scout' models on a balanced subset of the gated survivors
    to identify hard negatives.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.seed = config["settings"]["seed"]

    def _prepare_scout_data(self, X, y):
        """
        Creates a balanced 1:1 dataset for scout training.
        """
        seed_everything(self.seed)

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        # Downsample negatives to match positives
        n_pos = len(pos_indices)
        if len(neg_indices) > n_pos:
            neg_indices_sampled = np.random.choice(
                neg_indices, size=n_pos, replace=False
            )
        else:
            neg_indices_sampled = neg_indices

        sample_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(sample_indices)

        return X.iloc[sample_indices], y.iloc[sample_indices]

    def train(self, X, y):
        """
        Trains both LGBM and XGB scouts.
        """
        logger.info("Preparing balanced dataset for Scout training...")
        X_scout, y_scout = self._prepare_scout_data(X, y)

        logger.info(f"Scout Training Data Shape: {X_scout.shape}")

        # Initialize Scouts
        scout_lgbm = LGBMWrapper(self.config)
        scout_xgb = XGBWrapper(self.config)

        # Train Scouts
        logger.info("Training Scout A (LGBM)...")
        scout_lgbm.fit(X_scout, y_scout)

        logger.info("Training Scout B (XGB)...")
        scout_xgb.fit(X_scout, y_scout)

        return scout_lgbm, scout_xgb


class HardNegativeMiner:
    """
    Identifies hard negatives using the Dual-Scout consensus.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.threshold = config["training"]["scout_hard_negative_threshold"]
        self.scout_trainer = ScoutTrainer(config)

    def _mining_worker(self, X, y):
        """
        Worker function to be wrapped by cache.
        Trains scouts, predicts on full set, and returns indices of hard negatives.
        """
        # 1. Train Scouts
        scout_lgbm, scout_xgb = self.scout_trainer.train(X, y)

        # 2. Inference on Full Gated Dataset
        logger.info("Running Scout Inference on full training set for mining...")
        preds_lgbm = scout_lgbm.predict(X)
        preds_xgb = scout_xgb.predict(X)

        # 3. Identify Hard Negatives
        # Condition: True Label is 0 AND (P_lgbm > thresh OR P_xgb > thresh)
        is_negative = y == 0
        is_hard = (preds_lgbm > self.threshold) | (preds_xgb > self.threshold)

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = X.index[hard_negative_mask].to_numpy()

        logger.info(
            f"Mined {len(hard_negative_indices)} hard negatives from {np.sum(is_negative)} total negatives."
        )

        # Save scout models for reference (optional, but good for debugging)
        model_dir = self.config["paths"]["model_dir"]
        scout_lgbm.save(os.path.join(model_dir, "scout_lgbm.joblib"))
        scout_xgb.save(os.path.join(model_dir, "scout_xgb.joblib"))

        return hard_negative_indices

    def mine(self, X, y, load_cached_data=True):
        """
        Orchestrates the mining process with caching.
        """
        cache_key = "hard_negative_indices"

        # Config subset relevant to mining
        mining_config = {
            "threshold": self.threshold,
            "seed": self.config["settings"]["seed"],
            "training_params": self.config["training"],
        }

        return process_with_cache(
            func=self._mining_worker,
            cache_key=cache_key,
            config_dict=mining_config,
            load_cached_data=load_cached_data,
            file_format="npy",
            X=X,
            y=y,
        )


class ExpertTrainer:
    """
    Trains the final Dual-Model Ensemble using the Anchored Mining dataset.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.anchor_ratio = config["training"]["expert_anchor_ratio"]
        self.seed = config["settings"]["seed"]

    def _construct_expert_dataset(self, X, y, hard_neg_indices):
        """
        Constructs the training set: Positives + Hard Negatives + Random Anchors.
        """
        seed_everything(self.seed)

        # 1. Positives
        pos_indices = np.where(y == 1)[0]

        # 2. Hard Negatives
        # Ensure indices are valid for the current X (which might be reset_index)
        # hard_neg_indices passed in are likely from the original X index.
        # We assume X passed here is the same dataframe object or has same index as mining.
        # To be safe, we intersect with current index.
        valid_hard_neg = np.intersect1d(hard_neg_indices, X.index)

        # 3. Anchors (Easy Negatives)
        # Negatives that are NOT hard negatives
        all_neg_indices = np.where(y == 0)[0]
        # Convert to set for fast lookup
        hard_neg_set = set(valid_hard_neg)

        # Filter potential anchors
        # Note: all_neg_indices are integer positions if y is numpy, or index labels if series.
        # Let's assume y is Series and we work with index labels.
        neg_indices_labels = y.index[y == 0].to_numpy()

        easy_neg_candidates = [
            idx for idx in neg_indices_labels if idx not in hard_neg_set
        ]
        easy_neg_candidates = np.array(easy_neg_candidates)

        # Sample Anchors (1:1 with Positives)
        n_anchors = int(len(pos_indices) * self.anchor_ratio)
        if len(easy_neg_candidates) > n_anchors:
            anchor_indices = np.random.choice(
                easy_neg_candidates, size=n_anchors, replace=False
            )
        else:
            anchor_indices = easy_neg_candidates

        logger.info(
            f"Dataset Composition: {len(pos_indices)} Positives, {len(valid_hard_neg)} Hard Negatives, {len(anchor_indices)} Anchors."
        )

        # Combine
        final_indices = np.concatenate([pos_indices, valid_hard_neg, anchor_indices])
        # Note: pos_indices from np.where on series return integer locs? No, if y is series, we need index.
        # Let's standardize on index labels.
        pos_indices_labels = y.index[y == 1].to_numpy()
        final_indices = np.concatenate(
            [pos_indices_labels, valid_hard_neg, anchor_indices]
        )

        np.random.shuffle(final_indices)

        return X.loc[final_indices], y.loc[final_indices]

    def fit(self, X_train, y_train, hard_neg_indices, X_val, y_val):
        """
        Trains the ensemble.
        """
        logger.info("Constructing Expert Dataset...")
        X_expert, y_expert = self._construct_expert_dataset(
            X_train, y_train, hard_neg_indices
        )

        logger.info(f"Expert Training Data Shape: {X_expert.shape}")

        ensemble = DualModelEnsemble(self.config)
        ensemble.fit(X_expert, y_expert, X_val, y_val)

        return ensemble


class TrainingCurriculum:
    """
    Orchestrates the end-to-end KADM-AE training pipeline.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.data_loader = DataLoader(config)
        self.miner = HardNegativeMiner(config)
        self.expert_trainer = ExpertTrainer(config)

    def optimize_threshold(self, ensemble, X_val, y_val):
        """
        Finds the decision threshold that maximizes MCC on the validation set.
        """
        logger.info("Optimizing decision threshold...")
        preds = ensemble.predict(X_val)

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred_bin = (preds > thresh).astype(int)
            score = matthews_corrcoef(y_val, y_pred_bin)
            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        logger.info(f"Best Threshold: {best_thresh:.4f}, Validation MCC: {best_mcc}")
        return best_thresh

    def run(self, load_cached_data=True):
        """
        Executes the curriculum.
        """
        logger.info("Starting Training Curriculum...")

        # 1. Load Data
        # We load Gated Survivors for Train and Val
        X_train, y_train, _ = self.data_loader.load_dataset(
            "train", apply_gating=True, load_cached_data=load_cached_data
        )
        X_val, y_val, _ = self.data_loader.load_dataset(
            "val", apply_gating=True, load_cached_data=load_cached_data
        )

        # 2. Mine Hard Negatives
        # This step handles its own caching
        hard_neg_indices = self.miner.mine(
            X_train, y_train, load_cached_data=load_cached_data
        )

        # 3. Train Expert Ensemble
        ensemble = self.expert_trainer.fit(
            X_train, y_train, hard_neg_indices, X_val, y_val
        )

        # 4. Save Ensemble
        ensemble.save()

        # 5. Optimize Threshold
        best_threshold = self.optimize_threshold(ensemble, X_val, y_val)

        # Save threshold
        thresh_path = os.path.join(
            self.config["paths"]["model_dir"], "best_threshold.npy"
        )
        np.save(thresh_path, np.array([best_threshold]))
        logger.info(f"Threshold saved to {thresh_path}")

        return ensemble, best_threshold
