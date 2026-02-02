import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import set_seed, timer


class HeptViewEnsemble:
    """
    Hept-View Temporal-Topology Stacking Ensemble.

    Architecture:
    - Level 1: 7 Base Learners across 4 Modalities (Lexical, Behavioral, Semantic, Metadata)
    - Level 2: Logistic Regression Meta-Learner
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.models = {}
        self.meta_learner = LogisticRegression(**Config.META_LR_PARAMS)
        self.model_names = [
            "lexical_bagger",
            "community_bagger",
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
            "metadata_anchor",
            "temporal_booster",
        ]

        # Mapping models to their specific feature views
        self.feature_map = {
            "lexical_bagger": "lexical",
            "community_bagger": "behavioral",
            "semantic_booster": "semantic",
            "semantic_gradient": "semantic",
            "semantic_bagger": "semantic",
            "metadata_anchor": "metadata",
            "temporal_booster": "metadata",
        }

    def _get_model_instance(self, name):
        """Factory method to instantiate fresh base learners."""
        if name == "lexical_bagger":
            return RandomForestClassifier(**Config.LEXICAL_RF_PARAMS)
        elif name == "community_bagger":
            return RandomForestClassifier(**Config.COMMUNITY_RF_PARAMS)
        elif name == "semantic_booster":
            return xgb.XGBClassifier(**Config.SEMANTIC_XGB_PARAMS)
        elif name == "semantic_gradient":
            return lgb.LGBMClassifier(**Config.SEMANTIC_LGBM_PARAMS)
        elif name == "semantic_bagger":
            return RandomForestClassifier(**Config.SEMANTIC_RF_PARAMS)
        elif name == "metadata_anchor":
            return LogisticRegression(**Config.METADATA_LR_PARAMS)
        elif name == "temporal_booster":
            return lgb.LGBMClassifier(**Config.TEMPORAL_LGBM_PARAMS)
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _get_data_for_model(self, data_dict, model_name, indices=None):
        """Extracts the specific feature view and optionally slices it."""
        view_name = self.feature_map[model_name]
        X = data_dict[view_name]

        if indices is not None:
            if scipy.sparse.issparse(X):
                return X[indices]
            else:
                return X[indices]
        return X

    def train_oof(self, train_data, y_train):
        """
        Performs 5-Fold CV to generate OOF predictions for the Meta-Learner.
        """
        print(f"Starting Level 1 OOF Generation ({Config.N_FOLDS} folds)...")

        n_samples = len(y_train)
        oof_preds = pd.DataFrame(index=np.arange(n_samples), columns=self.model_names)
        oof_preds[:] = 0.0

        kf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Iterate over folds
        for fold, (train_idx, val_idx) in enumerate(
            kf.split(np.zeros(n_samples), y_train)
        ):
            print(f"  Fold {fold + 1}/{Config.N_FOLDS}")

            y_tr_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            # Calculate scale_pos_weight for this fold
            n_pos = np.sum(y_tr_fold)
            n_neg = len(y_tr_fold) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            for name in self.model_names:
                model = self._get_model_instance(name)
                X_tr_fold = self._get_data_for_model(train_data, name, train_idx)
                X_val_fold = self._get_data_for_model(train_data, name, val_idx)

                # Train
                if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
                    # Apply dynamic scale_pos_weight for XGB
                    if isinstance(model, xgb.XGBClassifier):
                        model.set_params(scale_pos_weight=scale_pos_weight)

                    # Use fold validation set for early stopping
                    model.fit(
                        X_tr_fold,
                        y_tr_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                else:
                    model.fit(X_tr_fold, y_tr_fold)

                # Predict
                preds = model.predict_proba(X_val_fold)[:, 1]
                oof_preds.loc[val_idx, name] = preds

        # Print OOF scores
        print("\nLevel 1 OOF AUC Scores:")
        for name in self.model_names:
            auc = roc_auc_score(y_train, oof_preds[name])
            print(f"  {name}: {auc:.16f}")

        return oof_preds

    def train_meta(self, oof_preds, y_train):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        print("\nTraining Level 2 Meta-Learner...")
        self.meta_learner.fit(oof_preds, y_train)

        # Check coefficients
        print("Meta-Learner Coefficients:")
        for name, coef in zip(self.model_names, self.meta_learner.coef_[0]):
            print(f"  {name}: {coef:.4f}")

    def train_final(self, train_data, y_train, val_data, y_val):
        """
        Retrains Level 1 models for final submission using Validation-Guided protocols.
        """
        print("\nRetraining Level 1 Models for Final Submission...")

        # Calculate global scale_pos_weight for XGB
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        for name in self.model_names:
            print(f"  Retraining {name}...")
            model = self._get_model_instance(name)

            X_train = self._get_data_for_model(train_data, name)
            X_val = self._get_data_for_model(val_data, name)

            if isinstance(model, (RandomForestClassifier, LogisticRegression)):
                # Strategy: Concatenate Train + Val for stable learners
                if scipy.sparse.issparse(X_train):
                    X_full = scipy.sparse.vstack([X_train, X_val])
                else:
                    X_full = np.vstack([X_train, X_val])

                y_full = np.concatenate([y_train, y_val])
                model.fit(X_full, y_full)

            elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
                # Strategy: Train on Train, use Val for Early Stopping
                if isinstance(model, xgb.XGBClassifier):
                    model.set_params(scale_pos_weight=scale_pos_weight)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )

            self.models[name] = model

        print("Final retraining complete.")

    def predict(self, test_data):
        """
        Generates final predictions for the test set.
        """
        print("Generating Final Predictions...")

        # 1. Generate Level 1 Predictions
        l1_preds = pd.DataFrame(
            index=np.arange(test_data["metadata"].shape[0]), columns=self.model_names
        )

        for name in self.model_names:
            model = self.models[name]
            X_test = self._get_data_for_model(test_data, name)
            l1_preds[name] = model.predict_proba(X_test)[:, 1]

        # 2. Generate Level 2 Predictions
        final_probs = self.meta_learner.predict_proba(l1_preds)[:, 1]

        return final_probs
