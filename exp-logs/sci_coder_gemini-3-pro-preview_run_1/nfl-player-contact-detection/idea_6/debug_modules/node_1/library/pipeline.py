import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, compute_mcc
from library.data_loader import load_metadata
from library.feature_engine import FeatureProcessor
from library.model_zoo import LGBMWrapper, XGBWrapper
from library.sampler import DataSampler


class IHNMEPipeline:
    """
    Orchestrates the Iterative Hard-Negative Mining Ensemble (IHNME) workflow.
    Manages data preparation, multi-stage training (Scout -> Mining -> Expert),
    threshold optimization, and final inference.
    """

    def __init__(self):
        self.logger = setup_logger("ihnme_pipeline")
        self.feature_processor = FeatureProcessor()
        self.sampler = DataSampler()

        # Columns to exclude from feature set
        self.ignore_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "contact",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "p2_int",
        ]

    def _get_feature_cols(self, df):
        """Identifies feature columns by excluding metadata and target columns."""
        return [c for c in df.columns if c not in self.ignore_cols]

    def _add_is_ground_feature(self, df):
        """Adds the 'is_ground' binary feature."""
        if "nfl_player_id_2" in df.columns:
            df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)
        return df

    def prepare_data(self, split="train", load_cached_data=True):
        """
        Loads metadata and generates features for the specified split.
        Adds the 'is_ground' feature.
        """
        self.logger.info(f"Preparing data for split: {split}")

        # Load metadata
        meta_df = load_metadata(split)

        # Generate features (handles caching internally)
        df = self.feature_processor.generate_features(
            meta_df, split=split, load_cached_data=load_cached_data
        )

        # Add manual features
        df = self._add_is_ground_feature(df)

        return df

    def run_phase_1_scout(self, train_df, val_df, feature_cols):
        """
        Phase 1: Train the Scout model (Lightweight LightGBM) on a subset.
        """
        self.logger.info("--- Phase 1: Scout Training ---")

        # Create Scout Dataset
        scout_df = self.sampler.create_scout_dataset(train_df, target_col="contact")

        # Initialize Model
        scout_model = LGBMWrapper(Config.LGBM_SCOUT_PARAMS, model_name="lgbm_scout")

        # Train
        X_train = scout_df[feature_cols]
        y_train = scout_df["contact"]
        X_val = val_df[feature_cols]
        y_val = val_df["contact"]

        scout_model.train(X_train, y_train, X_val, y_val)
        scout_model.save()

        # Cleanup
        del scout_df, X_train, y_train
        gc.collect()

        return scout_model

    def run_phase_2_mining(self, scout_model, train_df, feature_cols):
        """
        Phase 2: Use Scout model to mine Hard Negatives from the full training set.
        """
        self.logger.info("--- Phase 2: Hard Negative Mining ---")

        # Mine hard negatives
        hard_negs_df = self.sampler.mine_hard_negatives(
            scout_model, train_df, feature_cols, target_col="contact"
        )

        return hard_negs_df

    def run_phase_3_expert(self, train_df, hard_negs_df, val_df, feature_cols):
        """
        Phase 3: Train Expert Ensemble (LGBM + XGB) on Expert Dataset.
        Expert Dataset = Positives + Hard Negatives + Random Negatives.
        """
        self.logger.info("--- Phase 3: Expert Training ---")

        # Create Expert Dataset
        expert_df = self.sampler.create_expert_dataset(
            train_df, hard_negs_df, target_col="contact"
        )

        X_train = expert_df[feature_cols]
        y_train = expert_df["contact"]
        X_val = val_df[feature_cols]
        y_val = val_df["contact"]

        # 1. Train LightGBM Expert
        self.logger.info("Training Expert LightGBM...")
        lgbm_expert = LGBMWrapper(Config.LGBM_EXPERT_PARAMS, model_name="lgbm_expert")
        lgbm_expert.train(X_train, y_train, X_val, y_val)
        lgbm_expert.save()

        # 2. Train XGBoost Expert
        self.logger.info("Training Expert XGBoost...")
        xgb_expert = XGBWrapper(Config.XGB_EXPERT_PARAMS, model_name="xgb_expert")
        xgb_expert.train(X_train, y_train, X_val, y_val)
        xgb_expert.save()

        # Cleanup
        del expert_df, X_train, y_train
        gc.collect()

        return [lgbm_expert, xgb_expert]

    def optimize_threshold(self, models, val_df, feature_cols):
        """
        Predicts on validation set using the ensemble and finds the optimal threshold.
        """
        self.logger.info("--- Threshold Optimization ---")

        X_val = val_df[feature_cols]
        y_val = val_df["contact"].values

        # Ensemble Prediction (Average)
        preds = np.zeros(len(X_val))
        for model in models:
            p = model.predict(X_val)
            preds += p
        preds /= len(models)

        # Grid Search for Threshold
        best_threshold = 0.5
        best_mcc = -1.0

        thresholds = np.arange(0.1, 0.95, 0.01)
        for thresh in thresholds:
            y_pred_bin = (preds > thresh).astype(int)
            mcc = compute_mcc(y_val, y_pred_bin)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        self.logger.info(f"Optimal Threshold: {best_threshold:.4f}")
        self.logger.info(f"Best Validation MCC: {best_mcc:.10f}")

        return best_threshold

    def generate_submission(self, models, best_threshold):
        """
        Generates predictions for the test set and saves the submission file.
        """
        self.logger.info("--- Inference & Submission ---")

        # Load Test Data
        test_df = self.prepare_data(split="test", load_cached_data=True)
        feature_cols = self._get_feature_cols(test_df)

        # Validate feature columns match training
        # (Simple check: ensure 'is_ground' and others exist)

        X_test = test_df[feature_cols]

        # Ensemble Prediction
        preds = np.zeros(len(X_test))
        for model in models:
            p = model.predict(X_test)
            preds += p
        preds /= len(models)

        # Apply Threshold
        predictions = (preds > best_threshold).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": test_df["contact_id"], "contact": predictions}
        )

        # Save
        save_path = Config.SUBMISSION_PATH
        submission.to_csv(save_path, index=False)
        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission shape: {submission.shape}")

        return submission

    def run(self):
        """
        Executes the full pipeline.
        """
        # 1. Prepare Data
        train_df = self.prepare_data(split="train")
        val_df = self.prepare_data(split="val")

        feature_cols = self._get_feature_cols(train_df)
        self.logger.info(f"Feature Columns ({len(feature_cols)}): {feature_cols}")

        # 2. Phase 1: Scout
        scout_model = self.run_phase_1_scout(train_df, val_df, feature_cols)

        # 3. Phase 2: Mining
        hard_negs_df = self.run_phase_2_mining(scout_model, train_df, feature_cols)

        # Free up memory: Scout model no longer needed for inference if we have hard negs
        del scout_model
        gc.collect()

        # 4. Phase 3: Expert
        expert_models = self.run_phase_3_expert(
            train_df, hard_negs_df, val_df, feature_cols
        )

        # 5. Optimization
        best_threshold = self.optimize_threshold(expert_models, val_df, feature_cols)

        # 6. Inference
        self.generate_submission(expert_models, best_threshold)

        self.logger.info("Pipeline execution completed successfully.")
