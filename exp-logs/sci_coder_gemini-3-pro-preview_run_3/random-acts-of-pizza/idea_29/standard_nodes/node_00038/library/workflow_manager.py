import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_and_clean_data
from library.feature_engineering import FeatureEngineer
from library.model_architecture import PentViewEnsemble


class WorkflowManager:
    """
    Orchestrates the Cross-Validation Bagging Inference strategy.
    Manages data loading, fold generation, model training, persistence, and inference.
    """

    def __init__(self):
        self.logger = setup_logger("workflow_manager")
        self.models_dir = os.path.join(Config.CACHE_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        set_seed(Config.SEED)

    def train_cv_bagging(self, debug_size=None, load_cached_data=True):
        """
        Executes 5-Fold Stratified CV. Fits FeatureEngineer and PentViewEnsemble per fold.
        Saves trained artifacts for inference.

        Args:
            debug_size (int, optional): Number of samples to use for debugging.
            load_cached_data (bool): Whether to load cleaned data from cache.
        """
        # Apply debug configuration if provided
        if debug_size is not None:
            Config.DEBUG_SAMPLE_SIZE = debug_size
            self.logger.info(
                f"Debug mode enabled. Sample size: {Config.DEBUG_SAMPLE_SIZE}"
            )

        self.logger.info("Starting Cross-Validation Bagging Training...")

        # 1. Load Data
        # We load both train and val splits defined in metadata and combine them
        # to perform our own Stratified K-Fold CV.
        train_df, val_df, _ = load_and_clean_data(load_cached_data=load_cached_data)

        # Concatenate for full CV
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        y = full_df[Config.TARGET_COL].values

        self.logger.info(f"Combined Training Data Shape: {full_df.shape}")

        # 2. Setup CV
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(full_df))
        fold_aucs = []

        # 3. CV Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y)):
            self.logger.info(f"\n{'='*20} Fold {fold + 1} / {Config.N_FOLDS} {'='*20}")

            # Split Data
            fold_train_df = full_df.iloc[train_idx].copy()
            fold_val_df = full_df.iloc[val_idx].copy()
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # --- Feature Engineering ---
            # Fit fresh FE on fold training data to prevent leakage
            self.logger.info("Fitting FeatureEngineer...")
            fe = FeatureEngineer()
            fe.fit(fold_train_df)

            # Transform
            # We use fold-specific split names to ensure cache uniqueness (e.g. embeddings)
            self.logger.info("Transforming Fold Training Views...")
            train_views = fe.transform(
                fold_train_df,
                split_name=f"fold_{fold}_train",
                load_cache=load_cached_data,
            )

            self.logger.info("Transforming Fold Validation Views...")
            val_views = fe.transform(
                fold_val_df, split_name=f"fold_{fold}_val", load_cache=load_cached_data
            )

            # --- Model Training ---
            self.logger.info("Training PentViewEnsemble...")
            model = PentViewEnsemble()
            model.fit(train_views, y_fold_train, val_views, y_fold_val)

            # --- Evaluation ---
            val_preds = model.predict_proba(val_views)
            oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_fold_val, val_preds)
            fold_aucs.append(fold_auc)
            # Print full precision as requested
            print(f"Fold {fold + 1} AUC: {fold_auc}")

            # --- Persistence ---
            # Save FE and Model for inference
            fe_path = os.path.join(self.models_dir, f"fe_fold_{fold}.joblib")
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")

            # Clear embedding model from FE before saving to save space/time (it lazy loads)
            # Note: We can't easily modify the class, so we just save as is.
            joblib.dump(fe, fe_path)
            joblib.dump(model, model_path)
            self.logger.info(f"Saved artifacts for Fold {fold + 1}")

        # 4. Overall Metrics
        total_auc = roc_auc_score(y, oof_preds)
        mean_auc = np.mean(fold_aucs)

        self.logger.info(f"\n{'='*20} CV Complete {'='*20}")
        print(f"Overall OOF AUC: {total_auc}")
        print(f"Mean Fold AUC: {mean_auc}")

    def predict_bagged_inference(self, debug_size=None, load_cached_data=True):
        """
        Loads saved models from all folds, predicts on test set, and averages results.
        Saves submission file.
        """
        if debug_size is not None:
            Config.DEBUG_SAMPLE_SIZE = debug_size

        self.logger.info("Starting Bagged Inference...")

        # 1. Load Test Data
        _, _, test_df = load_and_clean_data(load_cached_data=load_cached_data)
        self.logger.info(f"Test Data Shape: {test_df.shape}")

        bagged_preds = np.zeros(len(test_df))

        # 2. Inference Loop
        for fold in range(Config.N_FOLDS):
            self.logger.info(f"Processing Fold {fold + 1}...")

            # Load Artifacts
            fe_path = os.path.join(self.models_dir, f"fe_fold_{fold}.joblib")
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")

            if not os.path.exists(fe_path) or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Artifacts for fold {fold} missing. Run train_cv_bagging first."
                )

            fe = joblib.load(fe_path)
            model = joblib.load(model_path)

            # Transform Test Data
            # Use fold-specific cache name to avoid collisions if FE differs
            test_views = fe.transform(
                test_df, split_name=f"fold_{fold}_test", load_cache=load_cached_data
            )

            # Predict
            preds = model.predict_proba(test_views)
            bagged_preds += preds

        # 3. Average Predictions
        avg_preds = bagged_preds / Config.N_FOLDS

        # 4. Save Submission
        submission = pd.DataFrame(
            {"request_id": test_df[Config.ID_COL], Config.TARGET_COL: avg_preds}
        )

        submission.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH}")
