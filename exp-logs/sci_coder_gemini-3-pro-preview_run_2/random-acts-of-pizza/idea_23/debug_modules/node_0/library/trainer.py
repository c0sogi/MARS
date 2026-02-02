import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import DataLoader
from library.feature_generator import FeatureGenerator
from library.custom_ensemble import StratifiedRandomSubspaceEnsemble


class Trainer:
    """
    Orchestrates the training pipeline for the Stratified Random Subspace Linear Ensemble.
    Handles Cross-Validation, Hyperparameter Tuning, and Submission Generation.
    """

    def __init__(self):
        self.logger = setup_logger("Trainer")
        self.feature_gen = FeatureGenerator()
        self.data_loader = DataLoader()

    def run_cross_validation(self):
        """
        Executes the full 5-fold stratified cross-validation pipeline.
        """
        self.logger.info("Starting Cross-Validation Pipeline...")

        # 1. Load Data
        # We load the predefined splits but will combine train/val for full CV
        df_train, df_val, df_test = self.data_loader.load_data(load_cached_data=True)

        # Combine train and val for full cross-validation to maximize data usage
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

        # 2. Feature Generation (Embeddings & Tabular)
        # Process splits separately to utilize caching effectively
        emb_train = self.feature_gen.generate_embeddings(df_train, "train")
        emb_val = self.feature_gen.generate_embeddings(df_val, "val")
        emb_test = self.feature_gen.generate_embeddings(df_test, "test")

        # Combine embeddings for full training set
        X_text_full = np.vstack([emb_train, emb_val])
        X_text_test = emb_test

        # Extract Tabular Features
        tab_train = self.feature_gen.extract_tabular_features(df_train)
        tab_val = self.feature_gen.extract_tabular_features(df_val)
        tab_test = self.feature_gen.extract_tabular_features(df_test)

        # Combine tabular features for full training set
        X_tab_full = np.vstack([tab_train, tab_val])
        X_tab_test = tab_test

        # Extract Target
        y_full = df_full[Config.TARGET_COL].values

        # 3. Cross-Validation Loop
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(df_full))
        test_preds_accum = np.zeros((len(df_test), Config.N_FOLDS))
        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_full, y_full)):
            self.logger.info(f"=== Fold {fold + 1}/{Config.N_FOLDS} ===")

            # Split Data for this Fold
            X_text_train, X_text_val = X_text_full[train_idx], X_text_full[val_idx]
            X_tab_train, X_tab_val = X_tab_full[train_idx], X_tab_full[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]

            # 4. Preprocessing (QuantileTransformer)
            # Fit scaler ONLY on training data of this fold to prevent leakage
            scaler = QuantileTransformer(
                output_distribution=Config.QUANTILE_OUTPUT_DIST,
                random_state=Config.SEED,
            )
            X_tab_train_scaled = scaler.fit_transform(X_tab_train)
            X_tab_val_scaled = scaler.transform(X_tab_val)
            X_tab_test_scaled = scaler.transform(X_tab_test)

            # 5. Hyperparameter Tuning (Grid Search)
            # Tune C and class_weight using a hold-out split within the training fold
            best_params = self._tune_hyperparameters(
                X_text_train, X_tab_train_scaled, y_train
            )
            self.logger.info(f"Best Params for Fold {fold + 1}: {best_params}")

            # 6. Train Final Model for Fold
            model = StratifiedRandomSubspaceEnsemble(
                n_estimators=Config.N_ESTIMATORS,
                subspace_fraction=Config.SUBSPACE_FRACTION,
                C=best_params["C"],
                class_weight=best_params["class_weight"],
                random_state=Config.SEED,
                n_jobs=1,  # Sequential training within fold
                verbose=0,
            )
            model.fit(X_text_train, X_tab_train_scaled, y_train)

            # 7. Evaluate on Validation Fold
            val_probs = model.predict_proba(X_text_val, X_tab_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            fold_aucs.append(auc)
            self.logger.info(f"Fold {fold + 1} AUC: {auc}")

            oof_preds[val_idx] = val_probs

            # 8. Predict on Test Set
            test_probs = model.predict_proba(X_text_test, X_tab_test_scaled)[:, 1]
            test_preds_accum[:, fold] = test_probs

            # 9. Save Artifacts
            self._save_artifacts(model, scaler, fold)

        # 10. Overall Metrics
        overall_auc = roc_auc_score(y_full, oof_preds)
        avg_auc = np.mean(fold_aucs)
        self.logger.info(f"Overall OOF AUC: {overall_auc}")
        self.logger.info(f"Average Fold AUC: {avg_auc}")

        # 11. Generate Submission
        avg_test_preds = np.mean(test_preds_accum, axis=1)
        self._create_submission(df_test, avg_test_preds)

    def _tune_hyperparameters(self, X_text, X_tab, y):
        """
        Performs a simple grid search using a hold-out split from the training set
        to find the optimal Regularization (C) and Class Weight.
        """
        # Create internal split (80% train, 20% validation)
        X_text_tr, X_text_val, X_tab_tr, X_tab_val, y_tr, y_val = train_test_split(
            X_text, X_tab, y, test_size=0.2, stratify=y, random_state=Config.SEED
        )

        best_auc = -1
        best_params = {"C": 1.0, "class_weight": None}

        # Grid Search
        for C in Config.LR_C_GRID:
            for cw in Config.LR_CLASS_WEIGHTS:
                # Train ensemble with candidate parameters
                model = StratifiedRandomSubspaceEnsemble(
                    n_estimators=Config.N_ESTIMATORS,
                    subspace_fraction=Config.SUBSPACE_FRACTION,
                    C=C,
                    class_weight=cw,
                    random_state=Config.SEED,
                    n_jobs=1,
                )
                model.fit(X_text_tr, X_tab_tr, y_tr)

                # Evaluate
                probs = model.predict_proba(X_text_val, X_tab_val)[:, 1]
                auc = roc_auc_score(y_val, probs)

                if auc > best_auc:
                    best_auc = auc
                    best_params = {"C": C, "class_weight": cw}

        return best_params

    def _save_artifacts(self, model, scaler, fold):
        """
        Saves the trained model and fitted scaler for the fold.
        """
        models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(model, os.path.join(models_dir, f"model_fold_{fold}.joblib"))
        joblib.dump(scaler, os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))

    def _create_submission(self, df_test, preds):
        """
        Creates and saves the submission CSV file.
        """
        submission = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
