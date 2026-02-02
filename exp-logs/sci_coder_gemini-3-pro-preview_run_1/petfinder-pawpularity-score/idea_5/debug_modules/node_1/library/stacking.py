import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import setup_logger, compute_rmse
from library.feature_extraction import extract_features

logger = setup_logger("Stacking")


class RidgeStacker:
    """
    Implements the Tri-Paradigm Stacking Ensemble strategy.

    This class manages:
    1. Level-0: Training expert-specific Ridge regressors on (Image Embeddings + Metadata)
       using K-Fold Cross-Validation.
    2. Level-1: Training a Meta-Learner (Ridge) on the OOF predictions from Level-0.
    3. Inference: Aggregating predictions from all experts and folds.
    """

    def __init__(self):
        self.n_folds = Config.N_FOLDS
        self.alphas = Config.RIDGE_ALPHAS
        self.expert_names = list(Config.MODELS.keys())

        # Storage for trained artifacts
        # Structure: { 'expert_name': [ (model_fold_0, scaler_fold_0), ... ] }
        self.level0_models = {name: [] for name in self.expert_names}
        self.level1_model = None

    def fit_level0(self, features_map, meta_features, targets):
        """
        Trains Level-0 experts using K-Fold CV.

        Args:
            features_map (dict): Dictionary mapping expert names to feature arrays (N_samples, D_emb).
            meta_features (np.ndarray): Binary metadata features (N_samples, D_meta).
            targets (np.ndarray): Target values (N_samples,).

        Returns:
            pd.DataFrame: OOF predictions for each expert.
        """
        logger.info(f"Starting Level-0 Training with {self.n_folds}-Fold CV...")

        n_samples = len(targets)
        oof_preds = np.zeros((n_samples, len(self.expert_names)))
        oof_df = pd.DataFrame(oof_preds, columns=self.expert_names)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=Config.SEED)

        for i, expert in enumerate(self.expert_names):
            logger.info(f"Training Expert: {expert}")
            img_features = features_map[expert]

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(
                kf.split(img_features, targets)
            ):
                # 1. Split Data
                X_img_train, X_img_val = img_features[train_idx], img_features[val_idx]
                X_meta_train, X_meta_val = (
                    meta_features[train_idx],
                    meta_features[val_idx],
                )
                y_train, y_val = targets[train_idx], targets[val_idx]

                # 2. Concatenate Image + Meta
                # This implements the "Metadata Fusion" strategy
                X_train = np.hstack([X_img_train, X_meta_train])
                X_val = np.hstack([X_img_val, X_meta_val])

                # 3. Scale Features
                # StandardScaler is crucial for mixing high-dim embeddings and binary flags
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 4. Train RidgeCV
                # Uses efficient LOO-CV internally to pick best alpha from Config.RIDGE_ALPHAS
                # Note: 'neg_root_mean_squared_error' is preferred if available, else MSE.
                model = RidgeCV(
                    alphas=self.alphas, scoring="neg_root_mean_squared_error"
                )
                model.fit(X_train_scaled, y_train)

                # 5. Predict
                val_pred = model.predict(X_val_scaled)

                # 6. Store OOF
                oof_df.iloc[val_idx, i] = val_pred

                # 7. Store Model & Scaler for Inference
                self.level0_models[expert].append((model, scaler))

                # 8. Metric
                rmse = compute_rmse(y_val, val_pred)
                fold_scores.append(rmse)

            avg_rmse = np.mean(fold_scores)
            logger.info(f"  Expert {expert} Avg OOF RMSE: {avg_rmse:.10f}")

        return oof_df

    def fit_level1(self, oof_df, targets):
        """
        Trains the Level-1 Meta-Learner on OOF predictions.
        """
        logger.info("Training Level-1 Meta-Learner...")

        # Meta-learner is also RidgeCV
        self.level1_model = RidgeCV(
            alphas=self.alphas, scoring="neg_root_mean_squared_error"
        )

        X_meta = oof_df.values
        self.level1_model.fit(X_meta, targets)

        # Evaluate on the training set (OOF)
        # This represents the expected performance of the ensemble
        preds = self.level1_model.predict(X_meta)
        score = compute_rmse(targets, preds)

        logger.info(f"Level-1 Ensemble OOF RMSE: {score:.10f}")
        logger.info(
            f"Meta-Learner Coefficients: {dict(zip(self.expert_names, self.level1_model.coef_))}"
        )
        logger.info(f"Meta-Learner Best Alpha: {self.level1_model.alpha_}")

    def predict(self, features_map, meta_features):
        """
        Generates final predictions for test data.
        Aggregates predictions from all folds of all experts, then passes to meta-learner.

        Args:
            features_map (dict): Test image features.
            meta_features (np.ndarray): Test metadata features.

        Returns:
            np.ndarray: Final predictions.
        """
        n_samples = meta_features.shape[0]
        level0_preds = np.zeros((n_samples, len(self.expert_names)))

        for i, expert in enumerate(self.expert_names):
            img_features = features_map[expert]
            X = np.hstack([img_features, meta_features])

            # Average predictions across all K folds
            fold_preds = []
            for model, scaler in self.level0_models[expert]:
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
                fold_preds.append(pred)

            # Mean of fold predictions
            level0_preds[:, i] = np.mean(fold_preds, axis=0)

        # Level-1 Prediction
        final_preds = self.level1_model.predict(level0_preds)

        # Clip to valid range (1-100)
        final_preds = np.clip(final_preds, 1.0, 100.0)

        return final_preds


def run_stacking_workflow():
    """
    Orchestrates the full stacking pipeline:
    1. Loads features for Train and Validation sets.
    2. Merges them for maximum training data.
    3. Trains the RidgeStacker (Level-0 and Level-1).
    4. Loads Test features.
    5. Generates and saves submission.
    """
    logger.info("Initializing Stacking Workflow...")

    # =========================================================================
    # 1. Load and Merge Training Data
    # =========================================================================
    # We combine the provided 'train' (80%) and 'val' (20%) splits to train on 100% of data.
    modes = ["train", "val"]
    full_features = {k: [] for k in Config.MODELS.keys()}
    full_meta = []
    full_targets = []

    for mode in modes:
        path = (
            Config.TRAIN_METADATA_PATH if mode == "train" else Config.VAL_METADATA_PATH
        )

        # Temporary storage for this mode to ensure alignment
        mode_meta = None
        mode_targets = None

        for expert in Config.MODELS.keys():
            # Load features (uses caching internally)
            feats, meta, targets, _ = extract_features(
                model_name=expert, metadata_path=path, mode=mode, load_cached_data=True
            )
            full_features[expert].append(feats)

            # Meta and targets are identical across experts for the same mode
            if mode_meta is None:
                mode_meta = meta
                mode_targets = targets

        full_meta.append(mode_meta)
        full_targets.append(mode_targets)

    # Concatenate arrays
    X_features_map = {k: np.concatenate(v, axis=0) for k, v in full_features.items()}
    X_meta = np.concatenate(full_meta, axis=0)
    y_targets = np.concatenate(full_targets, axis=0)

    logger.info(f"Combined Training Set Size: {len(y_targets)}")

    # =========================================================================
    # 2. Train Stacker
    # =========================================================================
    stacker = RidgeStacker()

    # Train Level-0 (Experts)
    oof_df = stacker.fit_level0(X_features_map, X_meta, y_targets)

    # Train Level-1 (Meta-Learner)
    stacker.fit_level1(oof_df, y_targets)

    # =========================================================================
    # 3. Load Test Data
    # =========================================================================
    test_features = {}
    test_meta = None
    test_ids = None

    for expert in Config.MODELS.keys():
        feats, meta, targets, ids = extract_features(
            model_name=expert,
            metadata_path=Config.TEST_METADATA_PATH,
            mode="test",
            load_cached_data=True,
        )
        test_features[expert] = feats
        if test_meta is None:
            test_meta = meta
            test_ids = ids

    # =========================================================================
    # 4. Generate Submission
    # =========================================================================
    logger.info("Generating predictions for test set...")
    final_preds = stacker.predict(test_features, test_meta)

    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
    )

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission Head:\n{submission_df.head().to_string()}")
