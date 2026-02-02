import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import set_seed
from library.feature_extractor import FeatureEngine
from library.preprocessing import StreamProcessor
from library.models import Level1Estimators, StackingMetaLearner, generate_submission


class CrossValidator:
    """
    Orchestrates the Quad-Stream Semantic-Geometric Ensemble pipeline.
    Manages feature extraction, processing, stacking cross-validation,
    retraining, and submission generation.
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.stream_processor = StreamProcessor()
        self.level1_estimators = Level1Estimators()
        self.meta_learner = StackingMetaLearner()
        set_seed(Config.SEED)

    def train_and_predict(self, load_cached_data=True):
        """
        Runs the full pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load intermediate features from disk.
        """
        print("Starting Pipeline Execution...")

        # ==========================================
        # 1. Feature Extraction & Processing
        # ==========================================
        # Extract raw features from backbones (Swin, EffNet, DINO, CLIP)
        # Returns dicts with keys 'train', 'val', 'test'
        raw_feats, meta_feats, targets, ids = self.feature_engine.extract_features(
            load_cached_data=load_cached_data
        )

        # Process features: PCA compression per stream + Metadata scaling
        # Returns numpy arrays
        X_train_split, X_val_split, X_test = self.stream_processor.process_features(
            raw_feats, meta_feats, load_cached_data=load_cached_data
        )

        # Retrieve targets for splits
        y_train_split = targets["train"]
        y_val_split = targets["val"]

        # ==========================================
        # 2. Data Consolidation
        # ==========================================
        # Combine Train and Validation splits to maximize data for Cross-Validation.
        # The Level 1 Estimator handles K-Fold splitting internally.
        print("Consolidating Train and Validation sets for Stacking CV...")
        X_full = np.concatenate([X_train_split, X_val_split], axis=0)
        y_full = np.concatenate([y_train_split, y_val_split], axis=0)

        print(f"Combined Training Data Shape: {X_full.shape}")
        print(f"Combined Target Data Shape: {y_full.shape}")

        # ==========================================
        # 3. Level 1: Base Model Training (CV)
        # ==========================================
        # Generate Out-of-Fold predictions
        oof_preds = self.level1_estimators.get_oof_predictions(X_full, y_full)

        # ==========================================
        # 4. Level 2: Meta-Learner Training
        # ==========================================
        # Train Linear Regression on OOF predictions
        self.meta_learner.fit(oof_preds, y_full)

        # ==========================================
        # 5. Final Retraining
        # ==========================================
        # Retrain base models on the full dataset for inference
        self.level1_estimators.fit_all(X_full, y_full)

        # ==========================================
        # 6. Inference
        # ==========================================
        print("Generating predictions for Test set...")

        # Get base model predictions for test set
        base_test_preds = self.level1_estimators.predict(X_test)

        # Get final stacked predictions
        final_predictions = self.meta_learner.predict(base_test_preds)

        # ==========================================
        # 7. Submission
        # ==========================================
        test_ids = ids["test"]
        generate_submission(test_ids, final_predictions)

        print("Pipeline execution completed successfully.")
