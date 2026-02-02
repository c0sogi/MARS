import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import setup_logger, seed_everything, reduce_mem_usage
from library.data_loader import DataLoader
from library.feature_engineering_tracking import TrackingFeatureEngineer
from library.feature_engineering_helmets import HelmetFeatureEngineer
from library.model_trainer import StreamModel


class EnsemblePipeline:
    """
    Orchestrates the Late-Fusion Multi-Modal Ensemble pipeline.
    Manages data loading, feature generation for two streams (Tracking & Helmets),
    model training, blending optimization, and final inference.
    """

    def __init__(self):
        self.config = Config
        self.logger = setup_logger("EnsemblePipeline")
        self.data_loader = DataLoader()

        # Feature Engineers
        self.fe_tracking = TrackingFeatureEngineer()
        self.fe_helmets = HelmetFeatureEngineer()

        # Models
        self.model_tracking = StreamModel(name="Model_StreamA_Tracking")
        self.model_helmets = StreamModel(name="Model_StreamB_Helmets")

        # Optimization Artifacts
        self.best_weight = 0.5
        self.best_threshold = 0.5
        self.best_val_mcc = -1.0

    def _get_feature_cols(self, df):
        """
        Identifies feature columns by excluding metadata columns.
        """
        metadata_cols = {
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
            "is_ground",
            "frame_approx",
        }
        # Also exclude any string columns just in case, though is_ground is int
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        # Ensure numeric
        numeric_cols = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )
        return numeric_cols

    def train(self, load_cached_data=True):
        """
        Executes the training pipeline:
        1. Load Metadata
        2. Generate/Load Features for Stream A (Tracking)
        3. Train Model A
        4. Generate/Load Features for Stream B (Helmets)
        5. Train Model B
        6. Optimize Blending Weight and Threshold
        """
        seed_everything(self.config.SEED)
        self.logger.info("Starting Ensemble Training Pipeline...")

        # 1. Load Metadata
        train_meta, val_meta, _ = self.data_loader.load_metadata()

        # =========================================================================
        # Stream A: Tracking (Kinematic)
        # =========================================================================
        self.logger.info("--- Stream A: Tracking Data ---")

        # Generate Features
        X_train_track = self.fe_tracking.create_features(
            train_meta, mode="train", load_cached_data=load_cached_data
        )
        X_val_track = self.fe_tracking.create_features(
            val_meta, mode="validation", load_cached_data=load_cached_data
        )

        # Identify Features
        track_features = self._get_feature_cols(X_train_track)
        self.logger.info(
            f"Stream A Features ({len(track_features)}): {track_features[:5]} ..."
        )

        # Train Model A
        self.model_tracking.train(
            X_train_track,
            X_val_track,
            feature_cols=track_features,
            target_col="contact",
        )

        # Get Validation Probabilities for Blending
        val_probs_a = self.model_tracking.predict(X_val_track, track_features)

        # Cleanup Stream A Training Data to free memory
        del X_train_track, X_val_track
        gc.collect()

        # =========================================================================
        # Stream B: Helmets (Visual-Geometric)
        # =========================================================================
        self.logger.info("--- Stream B: Helmet Data ---")

        # Generate Features
        X_train_helm = self.fe_helmets.create_features(
            train_meta, mode="train", load_cached_data=load_cached_data
        )
        X_val_helm = self.fe_helmets.create_features(
            val_meta, mode="validation", load_cached_data=load_cached_data
        )

        # Identify Features
        helm_features = self._get_feature_cols(X_train_helm)
        self.logger.info(
            f"Stream B Features ({len(helm_features)}): {helm_features[:5]} ..."
        )

        # Train Model B
        self.model_helmets.train(
            X_train_helm, X_val_helm, feature_cols=helm_features, target_col="contact"
        )

        # Get Validation Probabilities for Blending
        val_probs_b = self.model_helmets.predict(X_val_helm, helm_features)

        # Cleanup Stream B Training Data
        del X_train_helm, X_val_helm
        gc.collect()

        # =========================================================================
        # Optimization: Blending & Thresholding
        # =========================================================================
        self.logger.info("--- Ensemble Optimization ---")

        # Ground Truth
        y_true = val_meta["contact"].values

        # Optimize
        self.best_weight, self.best_threshold, self.best_val_mcc = (
            StreamModel.optimize_blending(
                y_true, val_probs_a, val_probs_b, n_trials=self.config.BLENDING_TRIALS
            )
        )

        self.logger.info(f"Optimization Complete.")
        self.logger.info(f"Best Weight (Stream A): {self.best_weight}")
        self.logger.info(f"Best Threshold: {self.best_threshold}")
        print(f"Final Optimized Validation MCC: {self.best_val_mcc}")

    def inference(self, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Load Test Metadata
        2. Generate Features for Stream A & B
        3. Predict Probabilities
        4. Blend and Threshold
        5. Save Submission
        """
        seed_everything(self.config.SEED)
        self.logger.info("Starting Ensemble Inference Pipeline...")

        # 1. Load Metadata
        _, _, test_meta = self.data_loader.load_metadata()

        # =========================================================================
        # Stream A Prediction
        # =========================================================================
        self.logger.info("Generating Stream A (Tracking) predictions...")
        X_test_track = self.fe_tracking.create_features(
            test_meta, mode="test", load_cached_data=load_cached_data
        )
        track_features = self._get_feature_cols(X_test_track)

        probs_a = self.model_tracking.predict(X_test_track, track_features)

        del X_test_track
        gc.collect()

        # =========================================================================
        # Stream B Prediction
        # =========================================================================
        self.logger.info("Generating Stream B (Helmet) predictions...")
        X_test_helm = self.fe_helmets.create_features(
            test_meta, mode="test", load_cached_data=load_cached_data
        )
        helm_features = self._get_feature_cols(X_test_helm)

        probs_b = self.model_helmets.predict(X_test_helm, helm_features)

        del X_test_helm
        gc.collect()

        # =========================================================================
        # Blending & Submission
        # =========================================================================
        self.logger.info("Blending predictions and applying threshold...")

        # Weighted Average
        final_probs = (self.best_weight * probs_a) + ((1 - self.best_weight) * probs_b)

        # Thresholding
        final_preds = (final_probs > self.best_threshold).astype(int)

        # Construct Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": test_meta["contact_id"], "contact": final_preds}
        )

        # Save
        save_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(save_path, index=False)

        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission shape: {submission.shape}")

        # Basic check
        pos_rate = submission["contact"].mean()
        self.logger.info(f"Predicted Positive Contact Rate: {pos_rate:.4f}")
