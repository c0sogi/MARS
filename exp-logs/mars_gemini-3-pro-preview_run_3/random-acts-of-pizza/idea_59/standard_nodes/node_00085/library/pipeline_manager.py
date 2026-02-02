import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    N_FOLDS,
    ID_COL,
    TARGET_COL,
)
from library.utils import get_logger, save_cache, load_cache, print_metrics
from library.models import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticGradient,
    SemanticBagger,
    MetadataAnchor,
    TemporalBooster,
    MetaLearner,
)

logger = get_logger("pipeline_manager")


class PipelineManager:
    def __init__(self):
        self.models_dir = os.path.join(WORKING_DIR, "models")
        self.cache_dir = os.path.join(WORKING_DIR, "cache")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define Model Groups for Hybrid Inference Protocol
        # Volatile: Gradient Boosting models (Use CV-Bagging)
        self.volatile_models = [
            SemanticBooster,
            SemanticGradient,
            TemporalBooster,
        ]

        # Stable: Bagging (RF) and Linear models (Use Full Retraining)
        self.stable_models = [
            LexicalBagger,
            CommunityBagger,
            SemanticBagger,
            MetadataAnchor,
        ]

        # Flatten list for iteration order consistency
        self.all_model_classes = self.volatile_models + self.stable_models

        # Map class names to instances for easy lookup later
        self.model_names = [cls().name for cls in self.all_model_classes]

    def _slice_features(self, features_dict, indices):
        """
        Slices a dictionary of features (numpy arrays or sparse matrices) based on indices.
        """
        sliced = {}
        for key, data in features_dict.items():
            if sp.issparse(data):
                sliced[key] = data[indices]
            else:
                sliced[key] = data[indices]
        return sliced

    def run_cv_and_oof(self, features_dict, y, load_cached_oof=True):
        """
        Runs 5-Fold Stratified CV to generate OOF predictions.
        Trains models per fold.
        """
        oof_file = "oof_predictions.npy"

        # Try loading OOF from cache
        if load_cached_oof:
            cached_oof = load_cache(oof_file, self.cache_dir)
            if cached_oof is not None:
                logger.info(f"Loaded OOF predictions from cache: {cached_oof.shape}")
                return cached_oof

        logger.info("Starting Cross-Validation and OOF Generation...")

        n_samples = len(y)
        n_models = len(self.all_model_classes)
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Iterate Folds
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y)
        ):
            logger.info(f"--- Fold {fold_idx + 1}/{N_FOLDS} ---")

            # Slice Data
            X_train = self._slice_features(features_dict, train_idx)
            y_train = y.iloc[train_idx]

            X_val = self._slice_features(features_dict, val_idx)
            y_val = y.iloc[val_idx]

            # Train each model
            for model_idx, ModelClass in enumerate(self.all_model_classes):
                model = ModelClass()
                model_name = model.name

                # Fit
                # Volatile models use X_val for early stopping inside fit()
                # Stable models ignore X_val inside fit() usually, but we pass it for consistency
                model.fit(X_train, y_train, X_val, y_val)

                # Predict OOF
                preds = model.predict_proba(X_val)
                oof_preds[val_idx, model_idx] = preds

                # Save Fold Model
                # We save ALL fold models.
                # Volatile needs them for CV-Bagging.
                # Stable technically doesn't need them for inference (we retrain full),
                # but we save them for reproducibility/debugging.
                fold_model_path = os.path.join(
                    self.models_dir, f"{model_name}_fold_{fold_idx}.joblib"
                )
                joblib.dump(model.model, fold_model_path)

        # Evaluate OOF Performance
        logger.info("--- OOF Performance ---")
        for i, name in enumerate(self.model_names):
            auc = roc_auc_score(y, oof_preds[:, i])
            print_metrics({f"{name} AUC": auc})

        # Cache OOF
        save_cache(oof_preds, oof_file, self.cache_dir)

        return oof_preds

    def train_meta_learner(self, oof_preds, y):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        logger.info("Training Meta-Learner...")
        meta = MetaLearner()
        meta.fit(oof_preds, y)
        meta.save(self.models_dir)

        # Check Meta-Learner fit on OOF (approximate performance)
        meta_preds = meta.predict_proba(oof_preds)
        auc = roc_auc_score(y, meta_preds)
        print_metrics({"Meta-Learner OOF AUC": auc})

        return meta

    def retrain_stable_full(self, features_dict, y):
        """
        Retrains 'Stable' models on the full Union Dataset.
        """
        logger.info("Retraining Stable Models on Full Union Dataset...")

        for ModelClass in self.stable_models:
            model = ModelClass()
            logger.info(f"Retraining {model.name} (Full)...")
            model.fit(features_dict, y)  # No validation set passed

            # Save with specific suffix
            path = os.path.join(self.models_dir, f"{model.name}_full.joblib")
            joblib.dump(model.model, path)

    def predict_and_submit(self, test_features, test_ids):
        """
        Generates predictions using the Hybrid Inference Protocol and creates submission file.
        """
        logger.info("Generating Final Predictions...")

        n_samples = len(test_ids)
        n_models = len(self.all_model_classes)
        level1_preds = np.zeros((n_samples, n_models))

        # 1. Generate Level 1 Predictions
        for i, ModelClass in enumerate(self.all_model_classes):
            temp_model = ModelClass()  # Just for name and helper methods
            name = temp_model.name

            # Check if Volatile or Stable logic applies
            is_volatile = any(issubclass(ModelClass, v) for v in self.volatile_models)

            if is_volatile:
                # CV-Bagging: Load all 5 fold models and average
                logger.info(f"Predicting {name} (Volatile: CV-Bagging)...")
                fold_preds = np.zeros(n_samples)
                for fold_idx in range(N_FOLDS):
                    path = os.path.join(
                        self.models_dir, f"{name}_fold_{fold_idx}.joblib"
                    )
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Fold model missing: {path}")

                    # Load into temp model wrapper
                    temp_model.model = joblib.load(path)
                    fold_preds += temp_model.predict_proba(test_features)

                level1_preds[:, i] = fold_preds / N_FOLDS

            else:
                # Stable: Load single full model
                logger.info(f"Predicting {name} (Stable: Full Model)...")
                path = os.path.join(self.models_dir, f"{name}_full.joblib")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Full model missing: {path}")

                temp_model.model = joblib.load(path)
                level1_preds[:, i] = temp_model.predict_proba(test_features)

        # 2. Generate Meta Predictions
        logger.info("Predicting Meta-Learner...")
        meta = MetaLearner()
        meta.load(self.models_dir)
        final_probs = meta.predict_proba(level1_preds)

        # 3. Create Submission
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {SUBMISSION_PATH}")
        logger.info(f"Submission Head:\n{submission.head()}")
