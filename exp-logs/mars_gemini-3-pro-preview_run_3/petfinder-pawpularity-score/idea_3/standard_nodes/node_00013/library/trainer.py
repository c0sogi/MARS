import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import setup_logger, compute_rmse
from library.feature_extractor import FeatureExtractor
from library.ensemble_model import Level1Predictors, MetaLearner


class CrossValidator:
    """
    Orchestrates the K-Fold Cross-Validation training of the Stacked Ensemble
    and generates the final submission.
    """

    def __init__(self):
        self.logger = setup_logger(name="trainer")
        self.feature_extractor = FeatureExtractor()

    def run(self):
        """
        Executes the full training pipeline:
        1. Feature Extraction (with caching)
        2. K-Fold CV to train Level 1 models and generate OOF predictions
        3. Training Level 2 Meta-Learner on OOF predictions
        4. Retraining Level 1 models on full data
        5. Generating and saving test predictions
        """
        # ==========================================
        # 1. Feature Extraction
        # ==========================================
        self.logger.info("Starting Feature Extraction...")
        # Load features (computes and caches if not present)
        data = self.feature_extractor.extract_and_cache(load_cached_data=True)

        # Merge Train and Val sets for K-Fold Cross-Validation
        # We merge them because we want to perform our own K-Fold split
        X = np.concatenate([data["train"]["features"], data["val"]["features"]], axis=0)
        y = np.concatenate([data["train"]["targets"], data["val"]["targets"]], axis=0)
        # IDs are needed mostly for debugging or if we wanted to save OOF with IDs,
        # but here we just need them to ensure alignment if needed.
        # train_ids = np.concatenate([data['train']['ids'], data['val']['ids']], axis=0)

        # Test Data
        X_test = data["test"]["features"]
        ids_test = data["test"]["ids"]

        self.logger.info(f"Combined Training Data Shape: {X.shape}")
        self.logger.info(f"Test Data Shape: {X_test.shape}")

        # ==========================================
        # 2. K-Fold Cross Validation (Level 1)
        # ==========================================
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        # Storage for Out-of-Fold (OOF) predictions
        # Level 1 output has 3 columns: [SVR, LGBM, Ridge]
        oof_preds = np.zeros((X.shape[0], 3))

        self.logger.info(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            self.logger.info(f"--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Initialize and Train Level 1 Predictors
            l1_model = Level1Predictors()
            l1_model.fit(X_train_fold, y_train_fold)

            # Predict on Validation Fold
            val_preds_l1 = l1_model.predict(X_val_fold)

            # Store OOF predictions
            oof_preds[val_idx] = val_preds_l1

            # Calculate and Log RMSE for each base model for this fold
            rmse_svr = compute_rmse(y_val_fold, val_preds_l1[:, 0])
            rmse_lgbm = compute_rmse(y_val_fold, val_preds_l1[:, 1])
            rmse_xgb = compute_rmse(y_val_fold, val_preds_l1[:, 2])

            self.logger.info(
                f"Fold {fold + 1} Level 1 RMSEs - SVR: {rmse_svr}, LGBM: {rmse_lgbm}, XGB: {rmse_xgb}"
            )

        # ==========================================
        # 3. Train Level 2 Meta-Learner
        # ==========================================
        self.logger.info("Training Level 2 Meta-Learner on OOF Predictions...")
        meta_learner = MetaLearner()
        meta_learner.fit(oof_preds, y)

        # Evaluate Overall OOF Performance
        oof_final_preds = meta_learner.predict(oof_preds)
        oof_rmse = compute_rmse(y, oof_final_preds)
        self.logger.info(f"Overall OOF RMSE with Meta-Learner: {oof_rmse}")

        # ==========================================
        # 4. Retrain Level 1 on Full Data
        # ==========================================
        self.logger.info("Retraining Level 1 Models on Full Dataset...")
        final_l1_model = Level1Predictors()
        final_l1_model.fit(X, y)

        # ==========================================
        # 5. Inference on Test Set
        # ==========================================
        self.logger.info("Generating Test Predictions...")

        # Get base predictions from retrained Level 1 models
        test_l1_preds = final_l1_model.predict(X_test)

        # Get final predictions from Meta-Learner
        final_test_preds = meta_learner.predict(test_l1_preds)

        # Clip predictions to valid range [1, 100]
        final_test_preds = np.clip(final_test_preds, 1.0, 100.0)

        # ==========================================
        # 6. Save Submission
        # ==========================================
        self.save_submission(ids_test, final_test_preds)

    def save_submission(self, ids, preds):
        """
        Saves the predictions to a CSV file in the required format.

        Args:
            ids (np.array): Array of test IDs.
            preds (np.array): Array of predicted Pawpularity scores.
        """
        submission_df = pd.DataFrame({"Id": ids, "Pawpularity": preds})

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Head of submission:\n{submission_df.head()}")
