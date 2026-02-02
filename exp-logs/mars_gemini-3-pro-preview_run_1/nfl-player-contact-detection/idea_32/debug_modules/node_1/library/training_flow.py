import os
import logging
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    setup_logging,
    seed_everything,
    calc_mcc,
    parameter_aware_cache,
)
from library.data_pipeline import DataPipeline
from library.models import TriEnsemble
from library.mining import ScoutMiner


class TrainingPipeline:
    """
    Orchestrates the Dual-Basis Time-Domain Anchored-Mining training curriculum.
    Manages the lifecycle of Scouts, Mining, Experts, and Inference.
    """

    def __init__(self, config=Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_pipeline = DataPipeline(config)

        # Define paths for model artifacts
        self.scout_model_paths = {
            "lgbm": self.config.MODEL_SCOUT_LGBM,
            "xgb": self.config.MODEL_SCOUT_XGB,
            "cat": self.config.MODEL_SCOUT_CAT,
        }
        self.expert_model_paths = {
            "lgbm": self.config.MODEL_EXPERT_LGBM,
            "xgb": self.config.MODEL_EXPERT_XGB,
            "cat": self.config.MODEL_EXPERT_CAT,
        }

    def train_scouts(self, df_train, load_cached_data=True):
        """
        Trains the Scout ensemble on a balanced dataset.
        Checks for cached models before training.
        """
        self.logger.info("--- Phase 1: Scout Training ---")
        ensemble = TriEnsemble(self.config)

        # Check if models exist and caching is requested
        models_exist = all(os.path.exists(p) for p in self.scout_model_paths.values())

        if load_cached_data and models_exist:
            self.logger.info("Loading cached Scout models...")
            ensemble.load_models(self.scout_model_paths)
        else:
            self.logger.info("Training Scout models from scratch...")
            # Construct Balanced Dataset
            df_scout = self.data_pipeline.construct_scout_dataset(df_train)

            # Train
            ensemble.fit(df_scout, df_val=None, model_names=["lgbm", "xgb", "cat"])

            # Save
            ensemble.save_models(self.scout_model_paths)

        return ensemble

    def run_mining_phase(self, df_train, scout_ensemble, load_cached_data=True):
        """
        Runs the Hard Negative Mining process using the Scout ensemble.
        Delegates to ScoutMiner which handles caching of indices.
        """
        self.logger.info("--- Phase 2: Hard Negative Mining ---")
        miner = ScoutMiner(self.config)

        # The miner.mine_hard_negatives method has the @parameter_aware_cache decorator
        hard_indices = miner.mine_hard_negatives(
            df_train, scout_ensemble, load_cached_data=load_cached_data
        )

        return hard_indices

    def train_experts(self, df_train, df_val, hard_indices, load_cached_data=True):
        """
        Trains the Expert ensemble on the Anchored dataset (Positives + Hard Negatives + Anchors).
        Checks for cached models before training.
        """
        self.logger.info("--- Phase 3: Expert Training ---")
        ensemble = TriEnsemble(self.config)

        # Check if models exist and caching is requested
        models_exist = all(os.path.exists(p) for p in self.expert_model_paths.values())

        if load_cached_data and models_exist:
            self.logger.info("Loading cached Expert models...")
            ensemble.load_models(self.expert_model_paths)
        else:
            self.logger.info("Training Expert models from scratch...")
            # Construct Anchored Dataset
            df_expert = self.data_pipeline.construct_expert_dataset(
                df_train, hard_indices
            )

            # Train (using validation set for early stopping)
            ensemble.fit(df_expert, df_val=df_val, model_names=["lgbm", "xgb", "cat"])

            # Save
            ensemble.save_models(self.expert_model_paths)

        return ensemble

    def optimize_threshold(self, expert_ensemble, df_val, load_cached_data=True):
        """
        Optimizes the decision threshold on the validation set to maximize MCC.
        Caches the best threshold to disk.
        """
        self.logger.info("--- Phase 4: Threshold Optimization ---")

        # Check cache for threshold
        if load_cached_data and os.path.exists(self.config.CACHE_BEST_THRESHOLD):
            best_threshold = np.load(self.config.CACHE_BEST_THRESHOLD).item()
            self.logger.info(f"Loaded cached best threshold: {best_threshold}")
            return best_threshold

        self.logger.info("Predicting on validation set...")
        y_val = df_val["contact"].values
        y_pred_proba = expert_ensemble.predict_proba(df_val)

        best_mcc = -1.0
        best_threshold = 0.5

        # Search space
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred_binary = (y_pred_proba > thresh).astype(int)
            score = calc_mcc(y_val, y_pred_binary)

            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        self.logger.info(f"Optimization Complete.")
        self.logger.info(f"Best MCC: {best_mcc}")  # Printing full precision
        self.logger.info(f"Best Threshold: {best_threshold}")

        # Save to cache
        try:
            os.makedirs(
                os.path.dirname(self.config.CACHE_BEST_THRESHOLD), exist_ok=True
            )
            np.save(self.config.CACHE_BEST_THRESHOLD, np.array(best_threshold))
        except Exception as e:
            self.logger.error(f"Failed to save threshold cache: {e}")

        return best_threshold

    def generate_submission(
        self, expert_ensemble, threshold, load_cached_data=True, debug=False
    ):
        """
        Generates the submission file for the test set.
        """
        self.logger.info("--- Phase 5: Submission Generation ---")

        # Load Test Data
        df_test = self.data_pipeline.load_data(
            mode="test", load_cached_data=load_cached_data, debug=debug
        )

        if df_test.empty:
            self.logger.warning("Test dataset is empty. Creating empty submission.")
            return

        # Predict
        self.logger.info("Generating predictions for test set...")
        probs = expert_ensemble.predict_proba(df_test)
        predictions = (probs > threshold).astype(int)

        # Prepare Submission DataFrame
        sub_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Ensure alignment with sample_submission.csv
        sample_path = self.config.SAMPLE_SUBMISSION_PATH
        if os.path.exists(sample_path):
            sample_sub = pd.read_csv(sample_path)
            # Merge to ensure all IDs are present and in order
            final_sub = sample_sub[["contact_id"]].merge(
                sub_df, on="contact_id", how="left"
            )
            # Fill missing (if any dropped during gating) with 0
            final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)
        else:
            final_sub = sub_df

        # Save
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        final_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {self.config.SUBMISSION_PATH}")

    def run(self, load_cached_data=True, debug=False):
        """
        Executes the full pipeline.
        """
        seed_everything(self.config.SEED)
        setup_logging()

        self.logger.info(f"Starting Training Pipeline (Debug={debug})")

        # 1. Load Data
        df_train = self.data_pipeline.load_data(
            mode="train", load_cached_data=load_cached_data, debug=debug
        )
        df_val = self.data_pipeline.load_data(
            mode="val", load_cached_data=load_cached_data, debug=debug
        )

        # 2. Train Scouts
        scouts = self.train_scouts(df_train, load_cached_data=load_cached_data)

        # 3. Mine Hard Negatives
        hard_indices = self.run_mining_phase(
            df_train, scouts, load_cached_data=load_cached_data
        )

        # 4. Train Experts
        experts = self.train_experts(
            df_train, df_val, hard_indices, load_cached_data=load_cached_data
        )

        # 5. Optimize Threshold
        best_threshold = self.optimize_threshold(
            experts, df_val, load_cached_data=load_cached_data
        )

        # 6. Generate Submission
        self.generate_submission(
            experts, best_threshold, load_cached_data=load_cached_data, debug=debug
        )

        self.logger.info("Pipeline execution completed successfully.")
