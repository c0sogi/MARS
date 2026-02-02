import os
import gc
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import setup_logger, compute_mcc, seed_everything, garbage_collection
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.model_factory import ModelFactory


class ExpertTrainer:
    """
    Manages the 'Expert' phase of the VRC-ME pipeline.
    Handles Tier 2 feature generation, heterogeneous ensemble training,
    threshold optimization, and final inference.
    """

    def __init__(self):
        self.logger = setup_logger(name="ExpertTrainer")
        self.loader = DataLoader()
        self.generator = FeatureGenerator()
        seed_everything(Config.SEED)

        # Artifact paths
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.lgbm_path = os.path.join(self.model_dir, "lgbm_expert.joblib")
        self.xgb_path = os.path.join(self.model_dir, "xgb_expert.joblib")
        self.threshold_path = os.path.join(self.model_dir, "threshold.joblib")

    def _get_features_and_target(self, df: pd.DataFrame):
        """
        Extracts Tier 2 features and target variable from the dataframe.
        """
        # Select only the features defined for Tier 2 in Config
        feature_cols = [c for c in Config.TIER2_FEATURES if c in df.columns]
        target = df["contact"] if "contact" in df.columns else None
        return df[feature_cols], target, feature_cols

    def train(self, mined_indices: np.ndarray, load_cached_data: bool = True):
        """
        Trains the Expert Ensemble (LGBM + XGB) on the mined subset.

        Args:
            mined_indices: Array of indices corresponding to the mined subset (Positives + Hard Negatives).
            load_cached_data: Whether to use cached feature datasets.
        """
        self.logger.info("Starting Expert Training Phase...")

        # ---------------------------------------------------------
        # 1. Generate Tier 2 Features for Training
        # ---------------------------------------------------------
        # We generate features for the full dataset first to ensure correct temporal windowing.
        merged_train = self.loader.get_merged_data(
            split="train", load_cached_data=load_cached_data
        )
        tracking_train = self.loader.load_tracking(split="train")

        self.logger.info("Generating Tier 2 features for Training set...")
        df_train_tier2 = self.generator.generate(
            merged_train,
            tracking_train,
            tier=2,
            split="train",
            load_cached_data=load_cached_data,
        )

        # Free memory
        del merged_train, tracking_train
        garbage_collection()

        # ---------------------------------------------------------
        # 2. Filter to Mined Subset
        # ---------------------------------------------------------
        self.logger.info(
            f"Filtering training data to {len(mined_indices)} mined samples..."
        )
        # Intersect to ensure indices exist (safety check)
        valid_indices = np.intersect1d(mined_indices, df_train_tier2.index)
        df_train_subset = df_train_tier2.loc[valid_indices].reset_index(drop=True)

        # We can drop the full train df now
        del df_train_tier2
        garbage_collection()

        # ---------------------------------------------------------
        # 3. Generate Tier 2 Features for Validation
        # ---------------------------------------------------------
        merged_val = self.loader.get_merged_data(
            split="val", load_cached_data=load_cached_data
        )
        tracking_val = self.loader.load_tracking(split="val")

        self.logger.info("Generating Tier 2 features for Validation set...")
        df_val_tier2 = self.generator.generate(
            merged_val,
            tracking_val,
            tier=2,
            split="val",
            load_cached_data=load_cached_data,
        )

        del merged_val, tracking_val
        garbage_collection()

        # ---------------------------------------------------------
        # 4. Prepare Feature Matrices
        # ---------------------------------------------------------
        X_train, y_train, features = self._get_features_and_target(df_train_subset)
        X_val, y_val, _ = self._get_features_and_target(df_val_tier2)

        self.logger.info(f"Training Data Shape: {X_train.shape}")
        self.logger.info(f"Validation Data Shape: {X_val.shape}")
        self.logger.info(f"Feature Count: {len(features)}")

        # ---------------------------------------------------------
        # 5. Train LightGBM Expert
        # ---------------------------------------------------------
        self.logger.info("Training LightGBM Expert...")
        lgbm_model = ModelFactory.create_model(stage="expert", model_type="lgbm")
        lgbm_model.fit(X_train, y_train, X_val, y_val)
        lgbm_model.save(self.lgbm_path)

        # ---------------------------------------------------------
        # 6. Train XGBoost Expert
        # ---------------------------------------------------------
        self.logger.info("Training XGBoost Expert...")
        xgb_model = ModelFactory.create_model(stage="expert", model_type="xgb")
        xgb_model.fit(X_train, y_train, X_val, y_val)
        xgb_model.save(self.xgb_path)

        # ---------------------------------------------------------
        # 7. Optimize Threshold (Ensemble)
        # ---------------------------------------------------------
        self.logger.info("Optimizing Ensemble Threshold on Validation Set...")

        p_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]
        p_ensemble = (p_lgbm + p_xgb) / 2.0

        best_mcc = -1.0
        best_thresh = 0.5

        # Grid search for threshold
        thresholds = np.arange(0.1, 0.91, 0.01)
        for t in thresholds:
            preds = (p_ensemble >= t).astype(int)
            score = compute_mcc(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        self.logger.info(
            f"Best Validation MCC: {best_mcc:.10f} at Threshold: {best_thresh:.2f}"
        )

        joblib.dump(best_thresh, self.threshold_path)

        # Cleanup
        del df_train_subset, df_val_tier2, X_train, X_val, y_train, y_val
        garbage_collection()

    def predict_test(self, load_cached_data: bool = True):
        """
        Generates predictions for the Test set using the trained Expert Ensemble.
        Saves the submission file.
        """
        self.logger.info("Starting Inference on Test Set...")

        # ---------------------------------------------------------
        # 1. Load Artifacts
        # ---------------------------------------------------------
        if not os.path.exists(self.lgbm_path) or not os.path.exists(self.xgb_path):
            raise FileNotFoundError(
                "Trained models not found. Please run train() first."
            )

        lgbm_model = ModelFactory.create_model(stage="expert", model_type="lgbm")
        lgbm_model.load(self.lgbm_path)

        xgb_model = ModelFactory.create_model(stage="expert", model_type="xgb")
        xgb_model.load(self.xgb_path)

        threshold = joblib.load(self.threshold_path)
        self.logger.info(f"Loaded decision threshold: {threshold}")

        # ---------------------------------------------------------
        # 2. Generate Tier 2 Features for Test
        # ---------------------------------------------------------
        merged_test = self.loader.get_merged_data(
            split="test", load_cached_data=load_cached_data
        )
        tracking_test = self.loader.load_tracking(split="test")

        self.logger.info("Generating Tier 2 features for Test set...")
        df_test_tier2 = self.generator.generate(
            merged_test,
            tracking_test,
            tier=2,
            split="test",
            load_cached_data=load_cached_data,
        )

        del merged_test, tracking_test
        garbage_collection()

        # ---------------------------------------------------------
        # 3. Ensemble Inference
        # ---------------------------------------------------------
        X_test, _, _ = self._get_features_and_target(df_test_tier2)

        self.logger.info("Predicting with LightGBM...")
        p_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

        self.logger.info("Predicting with XGBoost...")
        p_xgb = xgb_model.predict_proba(X_test)[:, 1]

        p_ensemble = (p_lgbm + p_xgb) / 2.0

        # Apply threshold
        predictions = (p_ensemble >= threshold).astype(int)

        # ---------------------------------------------------------
        # 4. Save Submission
        # ---------------------------------------------------------
        submission = df_test_tier2[["contact_id"]].copy()
        submission["contact"] = predictions

        save_path = Config.SUBMISSION_FILE
        submission.to_csv(save_path, index=False)

        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission Shape: {submission.shape}")

        # Final cleanup
        del df_test_tier2, X_test, p_lgbm, p_xgb, p_ensemble
        garbage_collection()
