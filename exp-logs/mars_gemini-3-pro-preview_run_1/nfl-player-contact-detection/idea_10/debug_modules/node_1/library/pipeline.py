import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, execute_with_cache, save_to_npy, load_from_npy
from library.data_manager import DataManager
from library.models import LGBMWrapper, XGBWrapper, Ensemble


class Pipeline:
    """
    Orchestrates the Ego-Centric Spatial Grid Mining Ensemble workflow.
    Manages the curriculum learning process: Scout Training -> Hard Negative Mining -> Expert Training.
    """

    def __init__(self):
        self.logger = setup_logger()
        self.dm = DataManager()
        self.ensemble = Ensemble()

    def run_scout_phase(self, load_cached_data=True):
        """
        Phase 1: Train the Scout Model on a balanced dataset to learn the general boundary.
        Returns the trained Scout model.
        """
        self.logger.info("\n=== Phase 1: Scout Training ===")

        scout_model = LGBMWrapper(name="scout_lgbm")

        # Check if model already exists to skip training
        if load_cached_data and scout_model.load():
            self.logger.info("Loaded pre-trained Scout model.")
            return scout_model

        # Load Data
        X_scout, y_scout = self.dm.get_scout_dataset(load_cached_data=load_cached_data)
        X_val, y_val, _ = self.dm.get_val_dataset(load_cached_data=load_cached_data)

        # Train
        scout_model.fit(X_scout, y_scout, X_val, y_val)
        scout_model.save()

        return scout_model

    def run_mining_phase(self, scout_model, load_cached_data=True):
        """
        Phase 2: Use Scout Model to mine Hard Negatives from the full gated training set.
        Returns a list/array of indices corresponding to hard negatives.
        """
        self.logger.info("\n=== Phase 2: Hard Negative Mining ===")
        cache_filename = "hard_negative_indices.npy"

        def _mine():
            self.logger.info("Mining hard negatives...")
            # Get full candidate set (X, y, meta)
            X_full, y_full, _ = self.dm.get_mining_candidates(
                load_cached_data=load_cached_data
            )

            # Predict
            probs = scout_model.predict(X_full)

            # Identify Hard Negatives: Ground Truth = 0 AND Prob > Threshold
            # We use numpy boolean indexing
            is_negative = (y_full == 0).values
            is_confusing = probs > Config.MINING_THRESHOLD

            hard_neg_mask = is_negative & is_confusing
            hard_neg_indices = np.where(hard_neg_mask)[0]

            self.logger.info(
                f"Mined {len(hard_neg_indices)} hard negatives from {len(X_full)} candidates."
            )
            return hard_neg_indices

        # Execute with caching
        hard_neg_indices = execute_with_cache(
            cache_filename, _mine, load_cached_data=load_cached_data
        )

        return hard_neg_indices

    def run_expert_phase(self, hard_neg_indices, load_cached_data=True):
        """
        Phase 3: Train the Expert Ensemble on Positives + Hard Negatives + Buffer.
        """
        self.logger.info("\n=== Phase 3: Expert Training ===")

        # Load Data
        X_expert, y_expert = self.dm.get_expert_dataset(
            hard_neg_indices, load_cached_data=load_cached_data
        )
        X_val, y_val, _ = self.dm.get_val_dataset(load_cached_data=load_cached_data)

        # Define Models
        lgbm_expert = LGBMWrapper(name="expert_lgbm")
        xgb_expert = XGBWrapper(name="expert_xgb")

        # Train LightGBM
        if not (load_cached_data and lgbm_expert.load()):
            lgbm_expert.fit(X_expert, y_expert, X_val, y_val)
            lgbm_expert.save()
        self.ensemble.add_model(lgbm_expert)

        # Train XGBoost
        if not (load_cached_data and xgb_expert.load()):
            xgb_expert.fit(X_expert, y_expert, X_val, y_val)
            xgb_expert.save()
        self.ensemble.add_model(xgb_expert)

    def optimize_threshold(self, load_cached_data=True):
        """
        Validates the ensemble and optimizes the decision threshold for MCC.
        """
        self.logger.info("\n=== Threshold Optimization ===")

        X_val, y_val, _ = self.dm.get_val_dataset(load_cached_data=load_cached_data)

        # Optimize
        best_thresh, best_mcc = self.ensemble.optimize_threshold(X_val, y_val)

        self.logger.info(f"Final Validation MCC: {best_mcc}")
        return best_thresh

    def run_inference(self, load_cached_data=True):
        """
        Generates predictions for the test set and creates the submission file.
        """
        self.logger.info("\n=== Inference & Submission ===")

        # Load Test Data
        X_test, meta_test = self.dm.get_test_dataset(load_cached_data=load_cached_data)

        # Ensure ensemble has threshold loaded (if not optimized in this run)
        if self.ensemble.best_threshold == 0.5:
            self.ensemble.load_threshold()

        # Predict
        self.logger.info("Predicting on Test Set...")
        preds = self.ensemble.predict(X_test)

        # Format Submission
        submission = meta_test[["contact_id"]].copy()
        submission["contact"] = preds

        # Save
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission.to_csv(save_path, index=False)
        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission shape: {submission.shape}")

    def execute(self, load_cached_data=True):
        """
        Main execution entry point.
        """
        # 1. Scout
        scout_model = self.run_scout_phase(load_cached_data=load_cached_data)

        # 2. Mine
        hard_neg_indices = self.run_mining_phase(
            scout_model, load_cached_data=load_cached_data
        )

        # 3. Expert
        self.run_expert_phase(hard_neg_indices, load_cached_data=load_cached_data)

        # 4. Optimize
        self.optimize_threshold(load_cached_data=load_cached_data)

        # 5. Inference
        self.run_inference(load_cached_data=load_cached_data)
