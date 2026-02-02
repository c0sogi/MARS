import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import (
    MODEL_DIR,
    SUBMISSION_PATH,
    N_FOLDS,
    RANDOM_SEED,
    MODEL_KEYS,
    VOLATILE_MODELS,
    STABLE_MODELS,
    EARLY_STOPPING_ROUNDS,
    ID_COL,
    TARGET_COL,
)
from library.utils import setup_logger, set_seed
from library.model_factory import ModelFactory

logger = setup_logger("trainer")


class Trainer:
    """
    Orchestrates the Symmetric Non-View Stacking Ensemble training pipeline.
    Implements the Hybrid Inference Protocol:
    - Volatile Learners: CV-Bagging (Train K models, Average predictions).
    - Stable Learners: Full-Retraining (Train K models for OOF, Retrain 1 model on full data).
    """

    def __init__(self):
        self.model_dir = MODEL_DIR
        self.submission_path = SUBMISSION_PATH
        self.n_folds = N_FOLDS
        self.random_seed = RANDOM_SEED
        set_seed(self.random_seed)

    def _prepare_input(self, features_dict, model_key, indices=None):
        """
        Constructs the specific feature matrix for a given model based on its branch.
        Concatenates modality-specific features with the Augmented Global Metadata.

        Args:
            features_dict (dict): Dictionary containing 'X_lexical', 'X_community', etc.
            model_key (str): The identifier of the model to determine feature mix.
            indices (np.array, optional): Indices to slice the data (for CV folds).

        Returns:
            scipy.sparse.csr_matrix or np.ndarray: The constructed feature matrix.
        """
        # Extract components
        X_lexical = features_dict["X_lexical"]
        X_community = features_dict["X_community"]
        X_semantic = features_dict["X_semantic"]
        X_metadata = features_dict["X_metadata"]

        # Slice if indices provided
        if indices is not None:
            X_lexical = X_lexical[indices]
            X_community = X_community[indices]
            X_semantic = X_semantic[indices]
            X_metadata = X_metadata[indices]

        # Feature Construction Logic
        if "lexical" in model_key:
            # Branch 1: Sparse Lexical + Metadata
            return sp.hstack([X_lexical, X_metadata], format="csr")

        elif "community" in model_key:
            # Branch 2: Sparse Behavioral + Metadata
            return sp.hstack([X_community, X_metadata], format="csr")

        elif "semantic" in model_key:
            # Branch 3: Dense Semantic + Metadata
            return np.hstack([X_semantic, X_metadata])

        elif "metadata" in model_key or "temporal" in model_key:
            # Branch 4: Metadata Only
            return X_metadata

        else:
            raise ValueError(
                f"Could not determine feature mapping for model: {model_key}"
            )

    def _save_model(self, model, filename):
        path = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)

    def _load_model(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def train_ensemble(self, train_features, test_features, test_ids):
        """
        Executes the full training and prediction pipeline.

        Args:
            train_features (dict): Dictionary of training features and target 'y'.
            test_features (dict): Dictionary of test features.
            test_ids (pd.Series or list): Request IDs for the test set (for submission).
        """
        y = train_features["y"]
        n_samples = len(y)
        n_test_samples = test_features["X_metadata"].shape[0]

        # Initialize OOF matrix and Test Prediction matrix
        # Columns correspond to MODEL_KEYS order
        oof_preds = pd.DataFrame(index=np.arange(n_samples), columns=MODEL_KEYS)
        test_preds_l1 = pd.DataFrame(
            index=np.arange(n_test_samples), columns=MODEL_KEYS
        )

        # ---------------------------------------------------------------------
        # Phase 1: Level 1 Base Learners (Cross-Validation)
        # ---------------------------------------------------------------------
        logger.info(f"Starting Level 1 Training with {self.n_folds}-Fold CV...")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_seed
        )

        # Iterate over each model type
        for model_key in MODEL_KEYS:
            logger.info(f"\nTraining Base Model: {model_key}")

            is_volatile = model_key in VOLATILE_MODELS
            model_oof = np.zeros(n_samples)

            # Storage for test predictions for this specific model (across folds for volatile)
            # For stable models, we will overwrite this later with the retrained model's pred
            fold_test_preds = np.zeros((n_test_samples, self.n_folds))

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(n_samples), y)
            ):
                # Prepare Data
                X_train = self._prepare_input(train_features, model_key, train_idx)
                y_train = y[train_idx]
                X_val = self._prepare_input(train_features, model_key, val_idx)
                y_val = y[val_idx]

                # Instantiate Model
                model = ModelFactory.get_model(model_key)

                # Fit
                if is_volatile:
                    # Volatile: Use Early Stopping
                    fit_params = {"eval_set": [(X_val, y_val)]}

                    model_type = str(type(model)).lower()

                    if "xgb" in model_type:
                        # XGBoost: Set early_stopping_rounds via set_params (Cite debug_lesson_10)
                        model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                        fit_params["verbose"] = False

                    elif "lgbm" in model_type:
                        # LightGBM: Use callbacks for early stopping (Cite debug_lesson_17)
                        callbacks = [
                            lgb.early_stopping(
                                stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                            ),
                            lgb.log_evaluation(period=0),
                        ]
                        fit_params["callbacks"] = callbacks

                    else:
                        fit_params["verbose"] = False

                    model.fit(X_train, y_train, **fit_params)

                    # Save Fold Model (Hybrid Inference: Keep all volatile models)
                    self._save_model(model, f"{model_key}_fold_{fold}.joblib")

                    # Predict on Test (Accumulate for Bagging)
                    X_test = self._prepare_input(test_features, model_key)
                    fold_test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

                else:
                    # Stable: Standard Fit
                    model.fit(X_train, y_train)
                    # We do NOT save fold models for stable learners (Hybrid Inference: Retrain later)

                # Predict OOF
                val_pred = model.predict_proba(X_val)[:, 1]
                model_oof[val_idx] = val_pred

                # Log Fold Score
                fold_auc = roc_auc_score(y_val, val_pred)
                # logger.info(f"  Fold {fold} AUC: {fold_auc:.10f}")

            # Store OOF
            oof_preds[model_key] = model_oof
            full_auc = roc_auc_score(y, model_oof)
            logger.info(f"  Full OOF AUC for {model_key}: {full_auc:.10f}")

            # Handle Test Predictions for Level 1
            if is_volatile:
                # Average across folds (CV-Bagging)
                test_preds_l1[model_key] = fold_test_preds.mean(axis=1)
            else:
                # Placeholder; Stable models will be retrained and predicted in Phase 2
                pass

        # ---------------------------------------------------------------------
        # Phase 2: Stable Learner Retraining (Full Union Dataset)
        # ---------------------------------------------------------------------
        logger.info("\nPhase 2: Retraining Stable Learners on Full Union Dataset...")

        for model_key in STABLE_MODELS:
            logger.info(f"Retraining {model_key}...")

            # Prepare Full Data
            X_full = self._prepare_input(train_features, model_key)
            y_full = y

            # Instantiate and Fit
            model = ModelFactory.get_model(model_key)
            model.fit(X_full, y_full)

            # Save Single Model
            self._save_model(model, f"{model_key}.joblib")

            # Generate Final Test Predictions
            X_test = self._prepare_input(test_features, model_key)
            test_preds_l1[model_key] = model.predict_proba(X_test)[:, 1]

        # ---------------------------------------------------------------------
        # Phase 3: Meta-Learner Training
        # ---------------------------------------------------------------------
        logger.info("\nPhase 3: Training Meta-Learner...")

        meta_learner = ModelFactory.get_model("meta_learner")
        meta_learner.fit(oof_preds.values, y)

        self._save_model(meta_learner, "meta_learner.joblib")

        meta_auc = roc_auc_score(y, meta_learner.predict_proba(oof_preds.values)[:, 1])
        logger.info(f"Meta-Learner CV AUC (on OOF): {meta_auc:.10f}")

        # ---------------------------------------------------------------------
        # Phase 4: Submission Generation
        # ---------------------------------------------------------------------
        logger.info("\nPhase 4: Generating Submission...")

        # Predict using Meta-Learner on Stacked Test Predictions
        final_probs = meta_learner.predict_proba(test_preds_l1.values)[:, 1]

        # Create Submission DataFrame
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

        # Save
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)

        logger.info(f"Submission saved to {self.submission_path}")
        logger.info("Training pipeline completed successfully.")
