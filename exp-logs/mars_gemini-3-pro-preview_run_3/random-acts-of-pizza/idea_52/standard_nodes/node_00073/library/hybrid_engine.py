import os
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import MODEL_DIR, SEED
from library.model_factory import get_learner
from library.utils import log_metric, set_seed


class HybridTrainer:
    """
    Implements the Hybrid Inference Protocol for training.
    - Volatile Learners (Boosting): Train K fold-models with Early Stopping. Persist all.
    - Stable Learners (Bagging/Linear): Train K fold-models for OOF only. Retrain 1 model on full data.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _slice_features(self, X_dict, indices):
        """
        Slices a dictionary of feature matrices (sparse or dense) based on indices.
        """
        return {key: val[indices] for key, val in X_dict.items()}

    def train_volatile(self, model_name, learner_name, X, y, folds):
        """
        Trains a volatile learner (e.g., XGB, LGBM) using K-Fold CV with Early Stopping.
        Saves ALL fold models.
        """
        set_seed(SEED)
        print(f"\nTraining Volatile Learner: {model_name} ({learner_name})")

        oof_preds = np.zeros(len(y))
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Slice data
            X_train = self._slice_features(X, train_idx)
            y_train = y.iloc[train_idx]
            X_val = self._slice_features(X, val_idx)
            y_val = y.iloc[val_idx]

            # Instantiate model
            model = get_learner(learner_name)

            # Fit with Early Stopping
            # Note: FeatureBinder handles the dictionary unpacking.
            # We pass eval_set as a list of tuples [(X_val, y_val)]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            # Predict
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            # Score
            score = roc_auc_score(y_val, val_preds)
            fold_scores.append(score)
            log_metric(f"{model_name}_fold_{fold_idx}_auc", score)

            # Save Fold Model
            save_path = os.path.join(
                self.model_dir, f"{model_name}_fold_{fold_idx}.joblib"
            )
            joblib.dump(model, save_path)

        avg_score = np.mean(fold_scores)
        log_metric(f"{model_name}_cv_auc", avg_score)
        return oof_preds

    def train_stable(
        self, model_name, learner_name, X, y, folds, X_retrain=None, y_retrain=None
    ):
        """
        Trains a stable learner (e.g., RF, LR).
        1. Performs CV to generate OOF predictions (models discarded).
        2. Retrains ONE model on X_retrain/y_retrain (or X/y if not provided) and saves it.
        """
        set_seed(SEED)
        print(f"\nTraining Stable Learner: {model_name} ({learner_name})")

        oof_preds = np.zeros(len(y))
        fold_scores = []

        # 1. Cross-Validation for OOF
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = self._slice_features(X, train_idx)
            y_train = y.iloc[train_idx]
            X_val = self._slice_features(X, val_idx)

            # Instantiate and Fit
            model = get_learner(learner_name)
            model.fit(X_train, y_train)

            # Predict
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            # Score
            score = roc_auc_score(y.iloc[val_idx], val_preds)
            fold_scores.append(score)
            log_metric(f"{model_name}_fold_{fold_idx}_auc", score)

            # We do NOT save fold models for stable learners to save space/time,
            # as we use the full-retrain strategy for inference.

        avg_score = np.mean(fold_scores)
        log_metric(f"{model_name}_cv_auc", avg_score)

        # 2. Full Retraining
        print(f"Retraining {model_name} on full dataset...")
        final_model = get_learner(learner_name)

        # Use provided retrain data (e.g., Train + Val) or fallback to CV data
        X_final = X_retrain if X_retrain is not None else X
        y_final = y_retrain if y_retrain is not None else y

        final_model.fit(X_final, y_final)

        # Save Single Model
        save_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(final_model, save_path)

        return oof_preds

    def train_meta(self, X_oof, y):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        set_seed(SEED)
        print("\nTraining Meta-Learner...")

        meta_model = get_learner("MetaLearner")
        meta_model.fit(X_oof, y)

        # Evaluate on training set (sanity check, though biased)
        train_preds = meta_model.predict_proba(X_oof)[:, 1]
        score = roc_auc_score(y, train_preds)
        log_metric("meta_learner_train_auc", score)

        save_path = os.path.join(self.model_dir, "meta_learner.joblib")
        joblib.dump(meta_model, save_path)

        return meta_model


class HybridPredictor:
    """
    Implements the Hybrid Inference Protocol for prediction.
    - Volatile: Loads K fold-models, averages predictions.
    - Stable: Loads 1 full model, predicts.
    - Meta: Stacks Level 1 predictions.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir

    def predict(self, X_test, volatile_models, stable_models):
        """
        Generates final predictions.

        Args:
            X_test (dict): Feature dictionary for test set.
            volatile_models (list): List of model names treated as volatile.
            stable_models (list): List of model names treated as stable.

        Returns:
            np.array: Final probabilities.
        """
        level1_preds = []

        # 1. Volatile Inference (Bagging Fold Models)
        for name in volatile_models:
            print(f"Predicting with Volatile Learner: {name}")
            fold_preds = []
            # Look for all fold files
            fold = 0
            while True:
                path = os.path.join(self.model_dir, f"{name}_fold_{fold}.joblib")
                if not os.path.exists(path):
                    break

                model = joblib.load(path)
                p = model.predict_proba(X_test)[:, 1]
                fold_preds.append(p)
                fold += 1

            if not fold_preds:
                raise FileNotFoundError(f"No fold models found for {name}")

            # Average predictions across folds
            avg_pred = np.mean(fold_preds, axis=0)
            level1_preds.append(avg_pred)

        # 2. Stable Inference (Single Full Model)
        for name in stable_models:
            print(f"Predicting with Stable Learner: {name}")
            path = os.path.join(self.model_dir, f"{name}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            model = joblib.load(path)
            p = model.predict_proba(X_test)[:, 1]
            level1_preds.append(p)

        # 3. Meta Inference
        # Stack features: shape (n_samples, n_models)
        X_meta = np.column_stack(level1_preds)

        print("Predicting with Meta-Learner...")
        meta_path = os.path.join(self.model_dir, "meta_learner.joblib")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta-learner not found: {meta_path}")

        meta_model = joblib.load(meta_path)
        final_preds = meta_model.predict_proba(X_meta)[:, 1]

        return final_preds
