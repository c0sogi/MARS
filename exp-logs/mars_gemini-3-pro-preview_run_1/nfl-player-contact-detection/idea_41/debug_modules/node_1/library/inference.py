import os
import numpy as np
import pandas as pd
import gc

from library.config import CACHE_DIR, SUBMISSION_FILE, SEED, TEST_METADATA_PATH
from library.utils import setup_logger, CacheManager, seed_everything
from library.data_processing import DataProcessor
from library.feature_engineering import FeatureEngineer
from library.model_definitions import LGBMExpert, XGBExpert


class InferencePipeline:
    """
    Manages the inference pipeline for the KAT-AME strategy.
    Loads trained Expert models, generates features for the test set,
    computes ensemble predictions, and generates the submission file.
    """

    def __init__(self, logger=None):
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "inference.log"))
        )
        self.cache_manager = CacheManager()
        self.model_dir = os.path.join(CACHE_DIR, "models")

        # Model paths
        self.lgbm_path = os.path.join(self.model_dir, "expert_lgbm.joblib")
        self.xgb_path = os.path.join(self.model_dir, "expert_xgb.joblib")
        self.threshold_path = os.path.join(self.model_dir, "best_threshold.npy")

    def run_inference(self, load_cached_data=True):
        """
        Executes the full inference process.

        Args:
            load_cached_data (bool): Whether to load test features from cache if available.
        """
        seed_everything(SEED)
        self.logger.info("Starting Inference Pipeline...")

        # =========================================================================
        # 1. Data Loading & Feature Engineering
        # =========================================================================
        processor = DataProcessor(logger=self.logger)
        engineer = FeatureEngineer(logger=self.logger)

        # Load Metadata and Tracking
        # We check if features are cached first inside create_features,
        # but we need to load raw data if cache is missing or load_cached_data is False.
        # However, FeatureEngineer.create_features handles the logic of loading from cache
        # OR computing from raw inputs. We just need to provide the raw inputs in case they are needed.

        # To avoid loading large CSVs unnecessarily, we can check cache existence manually
        # or just let the pipeline handle it. For robustness, we load the raw data pointers.
        # Note: DataProcessor.load_* reads the CSVs. To optimize, we could check cache first,
        # but FeatureEngineer.create_features requires the dataframes as input.
        # We will follow the standard flow: Load Data -> Create Features (which handles caching).

        self.logger.info("Loading Test Data...")
        df_meta_test = processor.load_metadata(split="test")
        df_tracking_test = processor.load_tracking(split="test")

        self.logger.info("Generating Test Features...")
        # split="test" ensures no gating is applied
        df_features_test = engineer.create_features(
            df_meta_test,
            df_tracking_test,
            split="test",
            load_cached_data=load_cached_data,
            save_output=True,
        )

        # Clean up raw data to free memory
        del df_meta_test, df_tracking_test
        gc.collect()

        # =========================================================================
        # 2. Prepare Features for Model
        # =========================================================================
        # Define metadata columns to exclude (must match training)
        exclude_cols = [
            "contact_id",
            "contact",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "nfl_player_id_2_numeric",  # Intermediate col if present
        ]

        # Select feature columns
        feature_cols = [c for c in df_features_test.columns if c not in exclude_cols]
        self.logger.info(f"Inference Feature Count: {len(feature_cols)}")

        X_test = df_features_test[feature_cols]
        contact_ids = df_features_test["contact_id"]

        # =========================================================================
        # 3. Load Models and Threshold
        # =========================================================================
        self.logger.info("Loading Trained Models...")

        if not os.path.exists(self.lgbm_path) or not os.path.exists(self.xgb_path):
            raise FileNotFoundError("Trained models not found. Run training first.")

        expert_lgbm = LGBMExpert(logger=self.logger)
        expert_lgbm.load(self.lgbm_path)

        expert_xgb = XGBExpert(logger=self.logger)
        expert_xgb.load(self.xgb_path)

        # Load Threshold
        if os.path.exists(self.threshold_path):
            best_threshold = np.load(self.threshold_path)[0]
            self.logger.info(f"Loaded optimized threshold: {best_threshold}")
        else:
            best_threshold = 0.5
            self.logger.warning(
                f"Threshold file not found. Defaulting to {best_threshold}"
            )

        # =========================================================================
        # 4. Generate Predictions
        # =========================================================================
        self.logger.info("Predicting with LGBM...")
        preds_lgbm = expert_lgbm.predict_proba(X_test)

        self.logger.info("Predicting with XGBoost...")
        preds_xgb = expert_xgb.predict_proba(X_test)

        # Ensemble (Unweighted Average)
        self.logger.info("Ensembling predictions...")
        preds_ensemble = (preds_lgbm + preds_xgb) / 2.0

        # Apply Threshold
        predictions_binary = (preds_ensemble >= best_threshold).astype(int)

        # =========================================================================
        # 5. Generate Submission File
        # =========================================================================
        self.logger.info("Generating submission file...")

        submission_df = pd.DataFrame(
            {"contact_id": contact_ids, "contact": predictions_binary}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

        # Save
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        self.logger.info(f"Submission saved to {SUBMISSION_FILE}")
        self.logger.info(f"Total Predictions: {len(submission_df)}")
        self.logger.info(f"Positive Predictions: {submission_df['contact'].sum()}")

        return submission_df
