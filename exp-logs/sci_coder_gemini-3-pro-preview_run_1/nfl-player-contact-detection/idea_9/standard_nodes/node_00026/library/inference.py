import pandas as pd
import numpy as np
import os
from library.config import (
    EXPERT_LGBM_PARAMS,
    EXPERT_XGB_PARAMS,
)
import library.config as config
from library.utils import setup_logging, CacheManager
from library.data_factory import DataFactory
from library.model_zoo import EnsembleModel


class InferenceManager:
    def __init__(self):
        """
        Initializes the inference manager.
        """
        self.logger = setup_logging()
        # Cache manager for loading the threshold from the training phase
        self.mining_cache = CacheManager(
            cache_dir=os.path.join(config.WORKING_DIR, "mining_cache")
        )
        self.data_factory = DataFactory(mode="test")

        # Columns to exclude from feature matrix X (must match MiningTrainer)
        self.ignore_cols = [
            "contact_id",  # Specific to test set
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

    def load_resources(self):
        """
        Loads the trained ensemble model and the optimized threshold.
        """
        self.logger.info("Loading resources for inference...")

        # 1. Load Ensemble Model
        ensemble = EnsembleModel(EXPERT_LGBM_PARAMS, EXPERT_XGB_PARAMS)
        if not ensemble.load():
            raise FileNotFoundError(
                "Trained models not found. Ensure training has completed successfully."
            )

        # 2. Load Threshold
        # The threshold is saved as a numpy array in the mining cache
        threshold_arr = self.mining_cache.load("best_threshold.npy")
        if threshold_arr is None:
            self.logger.warning("Optimized threshold not found. Defaulting to 0.5.")
            threshold = 0.5
        else:
            threshold = float(threshold_arr[0])

        self.logger.info(f"Loaded Threshold: {threshold}")
        return ensemble, threshold

    def generate_submission(self):
        """
        Executes the inference pipeline:
        1. Loads test features.
        2. Generates probabilities using the ensemble.
        3. Applies threshold.
        4. Saves submission.csv.
        """
        self.logger.info("Starting Submission Generation...")

        # 1. Load Data
        # get_test_dataset returns the full test set with features (no gating)
        df_test = self.data_factory.get_test_dataset()
        self.logger.info(f"Test Data Shape: {df_test.shape}")

        # 2. Prepare Features
        # Identify feature columns by excluding metadata
        feature_cols = [c for c in df_test.columns if c not in self.ignore_cols]
        X_test = df_test[feature_cols]

        # Ensure contact_id is available for submission
        if "contact_id" not in df_test.columns:
            raise KeyError("contact_id column missing from test dataset.")

        contact_ids = df_test["contact_id"]

        # 3. Load Model & Threshold
        ensemble, threshold = self.load_resources()

        # 4. Predict
        probs = ensemble.predict(X_test)

        # 5. Apply Threshold
        predictions = (probs >= threshold).astype(int)

        # 6. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": contact_ids, "contact": predictions}
        )

        # 7. Save
        self.logger.info(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        self.logger.info("Submission generated successfully.")
