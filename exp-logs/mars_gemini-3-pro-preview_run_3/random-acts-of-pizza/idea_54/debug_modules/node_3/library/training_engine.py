import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import CACHE_DIR, TARGET_COL, SEED
from library.model_factory import get_base_learners, get_meta_learner
from library.feature_engineering import FeatureFactory


class HybridTrainer:
    """
    Implements the Hybrid Training Protocol for the Oct-View Stacking Ensemble.
    - Volatile Learners (XGB/LGBM): CV-Bagging (Save 5 fold models), Early Stopping.
    - Stable Learners (RF/Linear): OOF for Meta, Full Retrain for Inference (Save 1 full model).
    """

    def __init__(self, feature_factory: FeatureFactory):
        self.feature_factory = feature_factory
        self.models_dir = os.path.join(CACHE_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Mapping models to their required feature subsets
        self.feature_map = {
            "lexical_bagger": ["lexical", "metadata"],
            "lexical_anchor": ["lexical", "metadata"],
            "community_bagger": ["behavioral", "metadata"],
            "semantic_booster": ["semantic", "metadata"],
            "semantic_gradient": ["semantic", "metadata"],
            "semantic_bagger": ["semantic", "metadata"],
            "metadata_anchor": ["metadata"],
            "temporal_booster": ["metadata"],
        }

    def train_level_1(self, df_full: pd.DataFrame, folds: list) -> pd.DataFrame:
        """
        Trains all Level 1 base learners.
        Returns a DataFrame containing OOF probability predictions for each model.
        """
        print("\n=== Starting Level 1 Training (Base Learners) ===")

        # Pre-compute all feature matrices for the full training set
        # We rely on FeatureFactory's caching to avoid re-computation
        print("Loading/Generating feature matrices...")
        feature_cache = self.feature_factory.transform(
            df_full, "full_train", load_cache=True
        )

        base_learners = get_base_learners()
        oof_preds = pd.DataFrame(index=df_full.index)
        y_full = df_full[TARGET_COL].values

        for model_name, model_instance in base_learners.items():
            print(f"\nTraining {model_name}...")

            # 1. Prepare Data
            required_keys = self.feature_map[model_name]
            X_full = self.feature_factory.combine_features(feature_cache, required_keys)

            # 2. Determine Learner Type
            is_volatile = isinstance(model_instance, (XGBClassifier, LGBMClassifier))
            learner_type = (
                "Volatile (CV-Bagging)" if is_volatile else "Stable (Full-Retrain)"
            )
            print(f"  Type: {learner_type}")

            # 3. Cross-Validation Loop (Generate OOF + Train Fold Models if Volatile)
            model_oof = np.zeros(len(df_full))
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train, y_train = X_full[train_idx], y_full[train_idx]
                X_val, y_val = X_full[val_idx], y_full[val_idx]

                # Clone model for this fold
                clf = clone(model_instance)

                # Train
                if is_volatile:
                    # Volatile: Use Early Stopping
                    eval_set = [(X_val, y_val)]
                    if isinstance(clf, XGBClassifier):
                        # XGBoost 1.6+ moved eval_metric to set_params/init
                        clf.set_params(eval_metric="auc")
                        clf.fit(X_train, y_train, eval_set=eval_set)
                    else:
                        # LightGBM accepts eval_metric in fit
                        clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc")
                    # Save Fold Model for Inference
                    self._save_model(clf, f"{model_name}_fold_{fold_idx}")
                else:
                    # Stable: Standard Fit
                    clf.fit(X_train, y_train)
                    # We do NOT save fold models for stable learners (we use full retrain later)

                # Predict OOF
                y_pred_prob = clf.predict_proba(X_val)[:, 1]
                model_oof[val_idx] = y_pred_prob

                # Score
                score = roc_auc_score(y_val, y_pred_prob)
                fold_scores.append(score)
                # print(f"    Fold {fold_idx}: AUC = {score:.6f}")

            avg_score = np.mean(fold_scores)
            print(f"  Average CV AUC: {avg_score:.10f}")

            # Store OOF predictions
            oof_preds[model_name] = model_oof

            # 4. Full Retraining (Stable Learners Only)
            if not is_volatile:
                print(f"  Retraining {model_name} on full dataset...")
                full_clf = clone(model_instance)
                full_clf.fit(X_full, y_full)
                self._save_model(full_clf, f"{model_name}_full")

        # Save OOF predictions to cache
        oof_path = os.path.join(CACHE_DIR, "oof_predictions.parquet")
        oof_preds.to_parquet(oof_path)
        print(f"Level 1 Training Complete. OOF predictions saved to {oof_path}")

        return oof_preds

    def train_level_2(self, oof_df: pd.DataFrame, y_true: np.ndarray):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        print("\n=== Starting Level 2 Training (Meta-Learner) ===")

        meta_learner = get_meta_learner()

        # Train on OOF predictions
        # oof_df columns match the keys in base_learners
        X_meta = oof_df.values

        meta_learner.fit(X_meta, y_true)

        # Evaluate In-Sample (just for sanity check, real eval is CV of L1)
        y_pred_meta = meta_learner.predict_proba(X_meta)[:, 1]
        score = roc_auc_score(y_true, y_pred_meta)
        print(f"Meta-Learner In-Sample AUC: {score:.10f}")

        # Print Coefficients
        print("Meta-Learner Coefficients:")
        for name, coef in zip(oof_df.columns, meta_learner.coef_[0]):
            print(f"  {name}: {coef:.4f}")

        self._save_model(meta_learner, "meta_learner")
        print("Level 2 Training Complete.")

    def _save_model(self, model, filename):
        """Saves a model to the models directory."""
        path = os.path.join(self.models_dir, f"{filename}.joblib")
        joblib.dump(model, path)
