import os
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.backbone_extractor import FeatureExtractor
from library.level0_experts import Level0Trainer
from library.feature_processor import create_interaction_matrix


class MetaLearner:
    """
    Orchestrates the Level-1 Stacking process.
    1. Aggregates Level-0 Expert Predictions (orchestrating feature extraction and L0 training if needed).
    2. Constructs Interaction-Aware Features (Preds + Meta + Preds*Meta).
    3. Trains a Bayesian Ridge Meta-Learner using CV.
    4. Generates final submission.
    """

    def __init__(self):
        self.config = Config
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)

        seed_everything(self.seed)

    def _load_metadata(self):
        """Loads train and test metadata."""
        train_df = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(self.config.VAL_METADATA_PATH)
        test_df = pd.read_csv(self.config.TEST_METADATA_PATH)

        # Concatenate Train and Val for full training set
        full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        return full_train_df, test_df

    def _get_backbone_list(self):
        """Returns list of backbone model names defined in Config."""
        return [
            self.config.MODEL_SIGLIP,
            self.config.MODEL_DINOV2,
            self.config.MODEL_CONVNEXT,
        ]

    def _get_expert_list(self):
        """Returns list of expert algorithms."""
        return ["ridge", "svr", "et", "lgbm"]

    def _collect_level0_predictions(self, full_train_df, test_df):
        """
        Iterates through backbones and experts to collect all L0 predictions.
        Returns aggregated prediction matrices.
        """
        backbones = self._get_backbone_list()
        experts = self._get_expert_list()

        # Containers for predictions
        # Key: f"{backbone}_{expert}" -> value: np.array
        train_preds_dict = {}
        test_preds_dict = {}

        # Initialize Extractors and Trainer
        l0_trainer = Level0Trainer()

        # We need targets for L0 training
        train_targets = full_train_df[self.config.TARGET_COL].values.astype(np.float32)

        for backbone in backbones:
            # 1. Extract Features for this backbone
            # We use the FeatureExtractor which handles caching
            extractor = FeatureExtractor(backbone)

            # Extract/Load Train+Val features
            # Note: We need to handle the split logic.
            # FeatureExtractor caches by split name ('train', 'val').
            # We need to merge them to match full_train_df order.

            # Load Train
            tr_feat, tr_ids, tr_meta, tr_targ = extractor.extract_and_cache(
                split_name="train", metadata_path=self.config.TRAIN_METADATA_PATH
            )
            # Load Val
            val_feat, val_ids, val_meta, val_targ = extractor.extract_and_cache(
                split_name="val", metadata_path=self.config.VAL_METADATA_PATH
            )

            # Merge to create Full Train inputs
            full_train_emb = np.concatenate([tr_feat, val_feat], axis=0)
            full_train_meta = np.concatenate([tr_meta, val_meta], axis=0)
            # Targets are already loaded from DF, but let's ensure consistency if needed.
            # We rely on the DF order which concatenates train then val.

            # Load Test
            test_feat, test_ids, test_meta, _ = extractor.extract_and_cache(
                split_name="test", metadata_path=self.config.TEST_METADATA_PATH
            )

            # 2. Train/Predict Experts
            for expert in experts:
                key = f"{backbone}_{expert}"

                oof, test_pred = l0_trainer.train_expert(
                    backbone_name=backbone.split("/")[
                        -1
                    ],  # Use short name for cache files
                    expert_name=expert,
                    train_embeddings=full_train_emb,
                    train_metadata=full_train_meta,
                    train_targets=train_targets,
                    test_embeddings=test_feat,
                    test_metadata=test_meta,
                    load_cached_data=True,
                )

                train_preds_dict[key] = oof
                test_preds_dict[key] = test_pred

        # Convert dicts to matrices (N_samples, N_experts)
        # Ensure consistent order of keys
        keys = sorted(train_preds_dict.keys())

        P_train = np.stack([train_preds_dict[k] for k in keys], axis=1)
        P_test = np.stack([test_preds_dict[k] for k in keys], axis=1)

        return P_train, P_test

    def _create_stratified_folds(self, targets, n_folds):
        """
        Creates Stratified K-Fold iterators by binning the continuous target.
        Same logic as Level0Trainer for consistency.
        """
        num_bins = 14
        if len(np.unique(targets)) < num_bins:
            bins = targets
        else:
            bins = pd.cut(targets, bins=num_bins, labels=False)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        return list(skf.split(np.zeros(len(targets)), bins))

    def train_predict(self, load_cached_data=True):
        """
        Main execution method.

        Args:
            load_cached_data (bool): If True, attempts to load intermediate L0 predictions from disk.
                                     (Passed down to Level0Trainer).
        """
        print("Step 1: Loading Metadata...")
        full_train_df, test_df = self._load_metadata()

        # Extract targets and binary metadata
        y_train = full_train_df[self.config.TARGET_COL].values.astype(np.float32)

        # Get binary metadata features for interaction terms
        meta_cols = self.config.META_FEATURES
        M_train = full_train_df[meta_cols].values.astype(np.float32)
        M_test = test_df[meta_cols].values.astype(np.float32)

        print("Step 2: Aggregating Level-0 Expert Predictions...")
        # P matrix shape: (N, n_experts)
        P_train, P_test = self._collect_level0_predictions(full_train_df, test_df)

        print(f"Collected predictions from {P_train.shape[1]} experts.")

        print("Step 3: Constructing Interaction-Aware Feature Matrices...")
        # Create X = [P, M, P*M]
        X_train = create_interaction_matrix(P_train, M_train)
        X_test = create_interaction_matrix(P_test, M_test)

        print(f"Meta-Feature Matrix Shape: {X_train.shape}")

        print("Step 4: Training Level-1 Bayesian Meta-Learner (CV)...")

        n_folds = self.config.N_FOLDS
        folds = self._create_stratified_folds(y_train, n_folds)

        meta_oof_preds = np.zeros(len(y_train))
        meta_test_preds_accum = np.zeros(len(test_df))

        # Bayesian Ridge with ARD
        # We use the params from config
        br_params = self.config.META_MODEL_PARAMS

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Split
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train
            model = BayesianRidge(**br_params)
            model.fit(X_tr, y_tr)

            # Predict
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            # Clip predictions to valid range
            val_pred = np.clip(val_pred, 1.0, 100.0)
            test_pred = np.clip(test_pred, 1.0, 100.0)

            # Store
            meta_oof_preds[val_idx] = val_pred
            meta_test_preds_accum += test_pred

            # fold_rmse = compute_rmse(y_val, val_pred)
            # print(f"  Meta-Fold {fold_idx+1} RMSE: {fold_rmse:.5f}")

        # Average test predictions
        final_test_preds = meta_test_preds_accum / n_folds

        # Calculate Final RMSE
        final_rmse = compute_rmse(y_train, meta_oof_preds)
        print(f"Final Ensemble CV RMSE: {final_rmse:.10f}")

        print("Step 5: Generating Submission...")
        submission = pd.DataFrame(
            {
                self.config.ID_COL: test_df[self.config.ID_COL],
                self.config.TARGET_COL: final_test_preds,
            }
        )

        submission.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")

        # Optional: Cache the meta-predictions
        np.save(os.path.join(self.working_dir, "meta_oof_preds.npy"), meta_oof_preds)
        np.save(os.path.join(self.working_dir, "meta_test_preds.npy"), final_test_preds)
