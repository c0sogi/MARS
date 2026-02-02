import os
import numpy as np
import pandas as pd
import logging
import joblib
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import (
    setup_logging,
    seed_everything,
    save_to_npy,
    load_from_npy,
    check_cache_exists,
)
from library.data_manager import DataManager
from library.model_factory import TriModelEnsemble

logger = logging.getLogger("NFL_Contact_Detection")


class TrainingManager:
    """
    Orchestrates the multi-stage training curriculum:
    1. Train Scouts on Balanced Data.
    2. Mine Hard Negatives using Scouts.
    3. Train Expert on Anchored Dataset (Pos + HardNegs + Anchors) with Label Smoothing.
    4. Optimize Threshold and Generate Submission.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.dm = DataManager(debug=debug)
        self.scout_dir = os.path.join(Config.WORKING_DIR, "models", "scouts")
        self.expert_dir = os.path.join(Config.WORKING_DIR, "models", "expert")

        os.makedirs(self.scout_dir, exist_ok=True)
        os.makedirs(self.expert_dir, exist_ok=True)

        # Ensure logging is set up if not already
        if not logger.hasHandlers():
            setup_logging()

    def train_scouts(self, force_retrain=False):
        """
        Phase 1: Train Scout models on a balanced subset of the gated survivors.
        """
        logger.info("--- Starting Phase 1: Scout Training ---")

        # Check if scouts already exist
        scout_exists = all(
            [
                os.path.exists(os.path.join(self.scout_dir, f"{name}_model.joblib"))
                for name in ["lgbm", "xgb", "hgb"]
            ]
        )

        if scout_exists and not force_retrain:
            logger.info("Scout models found. Skipping training.")
            return

        # Load Data
        train_df, val_df = self.dm.get_scout_data(load_cached_data=True)

        # Balance Training Data (1:1 Ratio)
        pos_df = train_df[train_df["contact"] == 1]
        neg_df = train_df[train_df["contact"] == 0]

        # Downsample negatives to match positives
        if len(neg_df) > len(pos_df):
            neg_df = neg_df.sample(n=len(pos_df), random_state=Config.SEED)

        balanced_train_df = (
            pd.concat([pos_df, neg_df])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        logger.info(
            f"Balanced Scout Train Size: {len(balanced_train_df)} (Pos: {len(pos_df)}, Neg: {len(neg_df)})"
        )

        # Prepare X, y
        X_train, y_train = self.dm.prepare_X_y(balanced_train_df)
        X_val, y_val = self.dm.prepare_X_y(val_df)

        # Train Ensemble
        ensemble = TriModelEnsemble()
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Save
        ensemble.save(self.scout_dir)
        logger.info("Phase 1 Complete: Scouts trained and saved.")

    def mine_hard_negatives(self, force_remine=False):
        """
        Phase 2: Use Scouts to identify Hard Negatives from the full training set.
        Hard Negative: Label=0 AND P(Contact) > Threshold (Union of Scouts).
        """
        logger.info("--- Starting Phase 2: Hard Negative Mining ---")

        indices_path = "hard_negative_indices.npy"
        if check_cache_exists(indices_path) and not force_remine:
            logger.info("Hard negative indices found in cache. Skipping mining.")
            return load_from_npy(indices_path)

        # Load Scouts
        ensemble = TriModelEnsemble()
        ensemble.load(self.scout_dir)

        # Load Full Training Data (Gated Survivors)
        # We need the full set, not balanced
        train_df, _ = self.dm.get_scout_data(load_cached_data=True)

        # Filter for Negatives only (we only mine negatives)
        # We keep the original index to map back later
        neg_mask = train_df["contact"] == 0
        neg_df = train_df[neg_mask]

        if neg_df.empty:
            logger.warning("No negatives found to mine.")
            empty_indices = np.array([])
            save_to_npy(empty_indices, indices_path)
            return empty_indices

        X_neg, _ = self.dm.prepare_X_y(neg_df)

        # Get predictions from all scouts
        preds_dict = ensemble.predict_individual(X_neg)

        # Compute Union: Max probability across scouts
        # Stack predictions: (N_samples, N_models)
        all_preds = np.vstack(list(preds_dict.values())).T
        max_preds = np.max(all_preds, axis=1)

        # Identify Hard Negatives
        hard_mask = max_preds > Config.HARD_NEGATIVE_THRESHOLD

        # Get the original indices from the dataframe
        hard_indices = neg_df.index[hard_mask].to_numpy()

        logger.info(
            f"Mined {len(hard_indices)} Hard Negatives out of {len(neg_df)} candidates."
        )

        save_to_npy(hard_indices, indices_path)
        return hard_indices

    def _apply_temporal_smoothing(self, df):
        """
        Applies Gaussian smoothing to the 'contact' label over time for each player pair.
        """
        logger.info("Applying Temporal Label Smoothing...")

        # Sort to ensure time continuity
        df = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Define smoothing function
        def smooth_labels(x):
            # Convert to float for smoothing
            return gaussian_filter1d(
                x.astype(float), sigma=Config.LABEL_SMOOTHING_SIGMA
            )

        # Apply per group
        # Note: We group by game_play and pair.
        # Ground interactions have nfl_player_id_2 = 'G' (string) or 0 (int if processed).
        # DataManager ensures IDs are consistent.

        # We create a new column 'soft_contact'
        df["soft_contact"] = df.groupby(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        )["contact"].transform(smooth_labels)

        return df

    def train_expert(self, force_retrain=False):
        """
        Phase 3: Train Expert Ensemble on Anchored Dataset with Soft Labels.
        """
        logger.info("--- Starting Phase 3: Expert Training ---")

        expert_exists = all(
            [
                os.path.exists(os.path.join(self.expert_dir, f"{name}_model.joblib"))
                for name in ["lgbm", "xgb", "hgb"]
            ]
        )

        if expert_exists and not force_retrain:
            logger.info("Expert models found. Skipping training.")
            return

        # 1. Load Hard Negatives
        hard_indices = load_from_npy("hard_negative_indices.npy")
        if hard_indices is None:
            logger.warning("Hard negative indices not found. Running mining...")
            hard_indices = self.mine_hard_negatives()

        # 2. Load Full Data
        train_df, val_df = self.dm.get_scout_data(load_cached_data=True)

        # 3. Apply Temporal Smoothing to create Soft Targets
        # We do this on the full train_df BEFORE subsetting for Expert Data
        # to ensure the smoothing has the full temporal context.
        train_df = self._apply_temporal_smoothing(train_df)

        # 4. Construct Expert Dataset (Pos + HardNeg + Anchors)
        # Note: get_expert_data uses 'contact' (binary) to identify Positives, which is correct.
        # It will preserve our new 'soft_contact' column.
        expert_df = self.dm.get_expert_data(train_df, hard_indices)

        # 5. Prepare Training Data (Soft Labels)
        X_train, y_train_soft = self.dm.prepare_X_y(
            expert_df, target_col="soft_contact"
        )

        # 6. Prepare Validation Data (Binary Labels for Metric Tracking)
        # We validate on the original binary labels to ensure we are optimizing for the competition metric.
        X_val, y_val_binary = self.dm.prepare_X_y(val_df, target_col="contact")

        # 7. Train Expert Ensemble
        ensemble = TriModelEnsemble()
        ensemble.fit(X_train, y_train_soft, X_val, y_val_binary)

        # 8. Save
        ensemble.save(self.expert_dir)
        logger.info("Phase 3 Complete: Expert models trained and saved.")

    def optimize_threshold(self):
        """
        Finds the optimal probability threshold maximizing MCC on the Validation set.
        """
        logger.info("--- Optimizing Decision Threshold ---")

        ensemble = TriModelEnsemble()
        ensemble.load(self.expert_dir)

        _, val_df = self.dm.get_scout_data(load_cached_data=True)
        X_val, y_val = self.dm.prepare_X_y(val_df)

        # Get Probabilities
        probs = ensemble.predict_proba(X_val)

        # Grid Search for Threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = matthews_corrcoef(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        logger.info(f"Best Threshold: {best_thresh:.4f} (Val MCC: {best_mcc:.6f})")

        # Save best threshold
        save_to_npy(np.array(best_thresh), "best_threshold.npy")
        return best_thresh

    def generate_submission(self):
        """
        Generates predictions for the Test set using the Expert models and optimal threshold.
        Saves to submission.csv.
        """
        logger.info("--- Generating Submission ---")

        # Load Model and Threshold
        ensemble = TriModelEnsemble()
        ensemble.load(self.expert_dir)

        thresh_arr = load_from_npy("best_threshold.npy")
        threshold = float(thresh_arr) if thresh_arr is not None else 0.5
        logger.info(f"Using Threshold: {threshold}")

        # Load Test Data
        test_df = self.dm.get_test_data(load_cached_data=True)
        X_test, _ = self.dm.prepare_X_y(test_df)

        # Predict
        probs = ensemble.predict_proba(X_test)
        predictions = (probs >= threshold).astype(int)

        # Format Submission
        submission = pd.DataFrame(
            {"contact_id": test_df["contact_id"], "contact": predictions}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(
            f"Submission saved to {Config.SUBMISSION_PATH} with {len(submission)} rows."
        )

    def run_pipeline(self):
        """
        Executes the full pipeline.
        """
        self.train_scouts()
        self.mine_hard_negatives()
        self.train_expert()
        self.optimize_threshold()
        self.generate_submission()
