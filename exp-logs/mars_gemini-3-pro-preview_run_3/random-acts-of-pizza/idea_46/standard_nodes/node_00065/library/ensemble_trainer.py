import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb

from library.config import Config
from library.utils import log_info, compute_auc, set_seed
from library.model_zoo import ModelZoo


class EnsembleTrainer:
    """
    Manages the training, cross-validation, and stacking of the Hex-View Ensemble.
    Implements Validation-Guided Retraining and Leakage-Robust Stacking.
    """

    def __init__(self):
        self.base_models = ModelZoo.get_base_models()
        self.meta_learner = ModelZoo.get_meta_learner()

        # Define feature requirements for all potential models
        all_requirements = {
            "Lexical_RF": ["lexical", "metadata"],
            "Community_RF": ["behavioral", "metadata"],
            "Semantic_XGB": ["semantic", "metadata"],
            "Semantic_RF": ["semantic", "metadata"],
            "Metadata_LR": ["metadata"],
            "Temporal_LGBM": ["metadata"],
        }

        # Filter requirements to only include active models
        self.model_requirements = {
            k: v for k, v in all_requirements.items() if k in self.base_models
        }

        # Store trained base models (from the final retraining phase)
        self.trained_base_models = {}

    def _get_combined_features(self, feature_data, split, view_names):
        """
        Retrieves and concatenates the specified feature views for a given split.
        Handles mixing of Sparse and Dense matrices.

        Args:
            feature_data (dict): Nested dict {view_name: {split: data}}.
            split (str): 'train', 'val', or 'test'.
            view_names (list): List of view keys to combine.

        Returns:
            Combined feature matrix (sparse or dense).
        """
        features_list = []
        is_sparse = False

        for view in view_names:
            if view not in feature_data:
                raise KeyError(f"View '{view}' not found in feature_data.")

            data = feature_data[view][split]

            # Check if sparse
            if sparse.issparse(data):
                is_sparse = True

            features_list.append(data)

        if not features_list:
            raise ValueError(f"No features found for views {view_names}")

        if len(features_list) == 1:
            return features_list[0]

        # Concatenation logic
        if is_sparse:
            # Convert all to sparse if any is sparse to allow hstack
            sparse_list = []
            for f in features_list:
                if sparse.issparse(f):
                    sparse_list.append(f)
                else:
                    sparse_list.append(sparse.csr_matrix(f))
            return sparse.hstack(sparse_list).tocsr()
        else:
            # All dense
            return np.hstack(features_list)

    def get_oof_predictions(self, feature_data, y_train):
        """
        Generates Out-of-Fold predictions for the training set using Stratified CV.
        These OOF predictions constitute the training data for the Level 2 Meta-Learner.

        Args:
            feature_data (dict): Dictionary containing all feature views.
            y_train (array-like): Target labels for the training set.

        Returns:
            pd.DataFrame: OOF predictions with shape (n_train, n_models).
        """
        set_seed(Config.RANDOM_SEED)

        # Ensure y_train is numpy array
        y_train = np.array(y_train)
        n_samples = len(y_train)
        model_names = list(self.base_models.keys())
        oof_preds = pd.DataFrame(index=np.arange(n_samples), columns=model_names)

        # Initialize K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        log_info(f"Starting Level 1 OOF Generation ({Config.N_FOLDS} folds)...")

        # Pre-assemble full training matrices for each model to avoid repeated stacking
        X_full_cache = {}
        for model_name, views in self.model_requirements.items():
            X_full_cache[model_name] = self._get_combined_features(
                feature_data, "train", views
            )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            log_info(f"  Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            y_fold_train = y_train[train_idx]

            for model_name, model_instance in self.base_models.items():
                # Clone model to ensure fresh start
                model = clone(model_instance)

                # Get data slice
                X_all = X_full_cache[model_name]
                X_fold_train = X_all[train_idx]
                X_fold_val = X_all[val_idx]

                # Disable early stopping for OOF generation as we don't use a separate eval set here
                # Cite debug_lesson_14
                if (
                    hasattr(model, "set_params")
                    and "early_stopping_rounds" in model.get_params()
                ):
                    model.set_params(early_stopping_rounds=None)

                # Fit model
                # Note: For OOF generation, we use standard fit without early stopping
                # to maintain consistency with standard stacking procedures,
                # reserving the validation set protocol for the final retraining.
                model.fit(X_fold_train, y_fold_train)

                # Predict
                if hasattr(model, "predict_proba"):
                    # Handle binary classification, take probability of class 1
                    preds = model.predict_proba(X_fold_val)[:, 1]
                else:
                    preds = model.predict(X_fold_val)

                oof_preds.loc[val_idx, model_name] = preds

        # Score OOF
        log_info("Level 1 OOF Scores:")
        for model_name in model_names:
            auc = compute_auc(y_train, oof_preds[model_name].values.astype(float))
            print(f"  {model_name}: AUC = {auc:.8f}")

        return oof_preds

    def train_meta_learner(self, oof_preds, y_train):
        """
        Trains the Level 2 Logistic Regression on the OOF predictions.

        Args:
            oof_preds (pd.DataFrame): OOF predictions from Level 1.
            y_train (array-like): Target labels.
        """
        log_info("Training Level 2 Meta-Learner...")
        self.meta_learner.fit(oof_preds, y_train)

        # Log coefficients for interpretability
        if hasattr(self.meta_learner, "coef_"):
            coefs = self.meta_learner.coef_[0]
            intercept = self.meta_learner.intercept_[0]
            log_info(f"Meta-Learner Intercept: {intercept:.4f}")
            for name, coef in zip(oof_preds.columns, coefs):
                print(f"  Weight for {name}: {coef:.4f}")

    def evaluate_on_holdout(self, feature_data, y_train, y_val):
        """
        Evaluates the ensemble on the hold-out validation set.

        Protocol:
        1. Train fresh Base Models on TRAIN only.
        2. Predict on VAL.
        3. Use the Meta-Learner (trained on OOF) to combine VAL predictions.
        4. Calculate AUC on VAL.

        This avoids leakage by not training on Val before evaluation.
        Cite debug_lesson_9: Rely on OOF Metrics when retraining on Validation Data (handled inside evaluate_on_holdout)
        """
        log_info("Evaluating Ensemble on Hold-Out Validation Set...")

        model_names = list(self.base_models.keys())
        n_val = len(y_val)
        l1_val_preds = pd.DataFrame(index=np.arange(n_val), columns=model_names)

        y_train = np.array(y_train)

        for model_name, model_instance in self.base_models.items():
            model = clone(model_instance)
            views = self.model_requirements[model_name]

            # Get Data
            X_train = self._get_combined_features(feature_data, "train", views)
            X_val = self._get_combined_features(feature_data, "val", views)

            # Train on Train
            # Note: We do not use early stopping with Val here to avoid leakage.
            model.fit(X_train, y_train)

            # Predict on Val
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)

            l1_val_preds[model_name] = preds

        # Meta-Learner Prediction
        # Uses the meta-learner already fitted on OOF predictions
        final_probs = self.meta_learner.predict_proba(l1_val_preds)[:, 1]

        score = compute_auc(y_val, final_probs)
        return score

    def retrain_final_models(self, feature_data, y_train, y_val):
        """
        Retrains all Level 1 base models using the Validation-Guided Protocol.
        - Tree Ensembles (RF) and Linear Models: Train on Train + Val.
        - Gradient Boosters (XGB/LGBM): Train on Train, use Val for Early Stopping.

        Args:
            feature_data (dict): Dictionary containing all feature views.
            y_train (array-like): Training labels.
            y_val (array-like): Validation labels.
        """
        log_info("Retraining Level 1 Base Models for Final Inference...")

        # Ensure arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        for model_name, model_instance in self.base_models.items():
            model = clone(model_instance)
            views = self.model_requirements[model_name]

            # Prepare Data
            X_train = self._get_combined_features(feature_data, "train", views)
            X_val = self._get_combined_features(feature_data, "val", views)

            # Determine Protocol
            is_boosting = "XGB" in model_name or "LGBM" in model_name

            if is_boosting:
                # Protocol: Train on Train, use Val for Early Stopping
                log_info(
                    f"  Retraining {model_name} with Early Stopping (Train | Val)..."
                )

                fit_params = {}
                if "XGB" in model_name:
                    # XGBoost: early_stopping_rounds is in __init__ (via Config),
                    # but eval_set is required in fit.
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["verbose"] = False
                elif "LGBM" in model_name:
                    # LightGBM: Pass callbacks for early stopping
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["eval_metric"] = "auc"
                    fit_params["callbacks"] = [
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]

                model.fit(X_train, y_train, **fit_params)

            else:
                # Protocol: Train on Train + Val (Maximize Data)
                log_info(f"  Retraining {model_name} on Full Data (Train + Val)...")

                # Combine data
                if sparse.issparse(X_train):
                    X_full = sparse.vstack([X_train, X_val])
                else:
                    X_full = np.vstack([X_train, X_val])

                y_full = np.concatenate([y_train, y_val])

                model.fit(X_full, y_full)

            self.trained_base_models[model_name] = model

    def generate_submission(self, feature_data, test_ids):
        """
        Generates predictions for the test set using the retrained base models
        and the trained meta-learner. Saves to CSV.

        Args:
            feature_data (dict): Dictionary containing all feature views.
            test_ids (array-like): IDs for the test set rows.

        Returns:
            pd.DataFrame: The submission DataFrame.
        """
        log_info("Generating Final Predictions on Test Set...")

        model_names = list(self.base_models.keys())
        n_test = len(test_ids)
        l1_test_preds = pd.DataFrame(index=np.arange(n_test), columns=model_names)

        # 1. Level 1 Predictions
        for model_name in model_names:
            if model_name not in self.trained_base_models:
                raise RuntimeError(f"Model {model_name} has not been retrained.")

            model = self.trained_base_models[model_name]
            views = self.model_requirements[model_name]
            X_test = self._get_combined_features(feature_data, "test", views)

            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_test)[:, 1]
            else:
                preds = model.predict(X_test)

            l1_test_preds[model_name] = preds

        # 2. Level 2 Prediction
        final_probs = self.meta_learner.predict_proba(l1_test_preds)[:, 1]

        # 3. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_probs}
        )

        # 4. Save
        log_info(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        return submission_df
