import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.models import ModelFactory


class StackingEngine:
    """
    Orchestrates the training of the Symmetric Multi-Modal Stacking Ensemble.
    Manages Level 1 (Base Learners) and Level 2 (Meta-Learner) training lifecycles.
    """

    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        # Define the order and keys for base models to ensure consistency between train/predict
        # Format: (Model Name, Feature Key, Factory Method Name, Is XGBoost)
        self.model_configs = [
            ("text_lexical", "lexical", "get_text_lexical_model", False),
            ("text_sem_xgb", "semantic", "get_text_semantic_models", True),
            ("text_sem_rf", "semantic", "get_text_semantic_models", False),
            ("behav_sparse", "community", "get_behavior_sparse_model", False),
            ("behav_dense", "persona", "get_behavior_dense_model", True),
            ("context", "meta", "get_context_model", False),
        ]

    def _get_scale_pos_weight(self, y):
        """Calculates scale_pos_weight for XGBoost based on class imbalance."""
        if len(y) == 0:
            return 1.0
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        return neg / pos if pos > 0 else 1.0

    def _instantiate_model(self, config_tuple, spw):
        """Helper to instantiate a model from the config tuple."""
        name, _, factory_method, _ = config_tuple

        factory_func = getattr(ModelFactory, factory_method)

        # Handle factory methods that return dictionaries or require args
        if factory_method == "get_text_semantic_models":
            models = factory_func(scale_pos_weight=spw)
            if "xgb" in name:
                return models["xgb"]
            else:
                return models["rf"]
        elif factory_method == "get_behavior_dense_model":
            return factory_func(scale_pos_weight=spw)
        else:
            return factory_func()

    def train_level_1_cv(self, X_dict, y):
        """
        Performs 5-Fold Stratified CV to generate OOF predictions for Level 2 training.
        """
        print(f"Starting Level 1 Cross-Validation ({Config.N_FOLDS} folds)...")

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        n_samples = len(y)
        n_models = len(self.model_configs)

        # Matrix to store OOF predictions [Samples, Models]
        oof_preds = np.zeros((n_samples, n_models))

        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            y_tr, y_va = y[train_idx], y[val_idx]
            spw = self._get_scale_pos_weight(y_tr)

            fold_pred_list = []

            for model_idx, config in enumerate(self.model_configs):
                name, feat_key, _, is_xgb = config

                # Instantiate fresh model
                model = self._instantiate_model(config, spw)

                # Get features
                X_feat = X_dict[feat_key]
                X_tr_feat = X_feat[train_idx]
                X_va_feat = X_feat[val_idx]

                # Fit
                if is_xgb:
                    # Use fold validation for early stopping to prevent gross overfitting during OOF generation
                    model.fit(
                        X_tr_feat,
                        y_tr,
                        eval_set=[(X_va_feat, y_va)],
                        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                else:
                    model.fit(X_tr_feat, y_tr)

                # Predict
                p = model.predict_proba(X_va_feat)[:, 1]
                oof_preds[val_idx, model_idx] = p
                fold_pred_list.append(p)

            # Evaluate Fold
            avg_p = np.mean(fold_pred_list, axis=0)
            score = roc_auc_score(y_va, avg_p)
            print(f"Fold {fold+1} Average AUC: {score}")
            fold_aucs.append(score)

        print(f"Level 1 CV Complete. Mean AUC: {np.mean(fold_aucs)}")
        return oof_preds

    def train_level_2(self, OOF_matrix, y):
        """
        Trains the Level 2 Meta-Learner (Logistic Regression) on OOF predictions.
        """
        print("Training Level 2 Meta-Learner...")
        self.meta_model = ModelFactory.get_meta_learner()
        self.meta_model.fit(OOF_matrix, y)

        # Evaluate on OOF (In-sample for Meta, but OOF for Base)
        preds = self.meta_model.predict_proba(OOF_matrix)[:, 1]
        score = roc_auc_score(y, preds)
        print(f"Level 2 Stacked OOF AUC: {score}")
        print(f"Meta-Learner Coefficients: {self.meta_model.coef_}")

    def retrain_level_1_final(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Retrains all base learners using the Validation-Guided protocol.
        - RF/LR: Train on concatenated Train + Val.
        - XGB: Train on Train, use Val for Early Stopping.
        """
        print("Retraining Level 1 Base Models (Validation-Guided)...")

        # Prepare concatenated data for non-XGB models
        X_full_dict = {}
        for k in X_train_dict:
            if sp.issparse(X_train_dict[k]):
                X_full_dict[k] = sp.vstack([X_train_dict[k], X_val_dict[k]])
            else:
                X_full_dict[k] = np.vstack([X_train_dict[k], X_val_dict[k]])

        y_full = np.concatenate([y_train, y_val])

        # Calculate SPW based on Training set (for XGB) and Full set (for RF/LR)
        spw_train = self._get_scale_pos_weight(y_train)
        spw_full = self._get_scale_pos_weight(y_full)

        for config in self.model_configs:
            name, feat_key, _, is_xgb = config
            print(f"Retraining {name}...")

            if is_xgb:
                # Validation-Guided: Train on Train, Stop on Val
                model = self._instantiate_model(config, spw_train)
                model.fit(
                    X_train_dict[feat_key],
                    y_train,
                    eval_set=[(X_val_dict[feat_key], y_val)],
                    early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                # Full Retraining: Train on Train + Val
                model = self._instantiate_model(config, spw_full)
                model.fit(X_full_dict[feat_key], y_full)

            self.base_models[name] = model

        # Save models
        print(f"Saving models to {Config.MODEL_DIR}...")
        joblib.dump(
            self.base_models, os.path.join(Config.MODEL_DIR, "base_models.joblib")
        )
        joblib.dump(
            self.meta_model, os.path.join(Config.MODEL_DIR, "meta_model.joblib")
        )

    def predict(self, X_dict):
        """
        Generates predictions for new data using the stacked ensemble.
        """
        n_samples = X_dict["meta"].shape[0]
        n_models = len(self.model_configs)
        base_preds = np.zeros((n_samples, n_models))

        for i, (name, feat_key, _, _) in enumerate(self.model_configs):
            if name not in self.base_models:
                raise ValueError(
                    f"Model {name} not found. Call retrain_level_1_final first."
                )

            model = self.base_models[name]
            base_preds[:, i] = model.predict_proba(X_dict[feat_key])[:, 1]

        final_preds = self.meta_model.predict_proba(base_preds)[:, 1]
        return final_preds
