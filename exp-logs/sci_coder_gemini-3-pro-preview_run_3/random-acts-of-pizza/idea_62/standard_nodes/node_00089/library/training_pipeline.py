import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_union_dataset
from library.feature_engineering import FeatureEngineeringPipeline
from library.model_factory import ModelFactory

logger = setup_logger("training_pipeline")


class TrainingPipeline:
    """
    Orchestrates the training of the Conservative Granular Hept-View Stacking Ensemble.
    Implements the 'Consistent Hybrid Inference' protocol:
    - Volatile Learners (XGB/LGBM): 5-Fold CV with Early Stopping, saving all fold models (CV-Bagging).
    - Stable Learners (RF/LR): 5-Fold CV for OOF generation, followed by a single full-data retrain for inference.
    """

    def __init__(self):
        self.model_dir = Config.MODEL_DIR
        self.oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_STATE

        # Define the 7 branches and their properties
        # Type: 'volatile' (CV-Bagging) or 'stable' (Full-Retrain)
        self.branches = {
            "lexical_bagger": {
                "type": "stable",
                "factory": ModelFactory.get_lexical_bagger,
            },
            "community_bagger": {
                "type": "stable",
                "factory": ModelFactory.get_community_bagger,
            },
            "semantic_booster": {
                "type": "volatile",
                "factory": ModelFactory.get_semantic_booster,
            },
            "semantic_gradient": {
                "type": "volatile",
                "factory": ModelFactory.get_semantic_gradient,
            },
            "semantic_bagger": {
                "type": "stable",
                "factory": ModelFactory.get_semantic_bagger,
            },
            "metadata_anchor": {
                "type": "stable",
                "factory": ModelFactory.get_metadata_anchor,
            },
            "temporal_booster": {
                "type": "volatile",
                "factory": ModelFactory.get_temporal_booster,
            },
        }

    def _prepare_features(self, features_dict, branch_name):
        """
        Constructs the specific feature set for a given branch by concatenating
        the primary view with the augmented global metadata.
        """
        X_meta = features_dict["train_meta"]

        if branch_name == "lexical_bagger":
            # Sparse Lexical + Dense Meta
            return sp.hstack([features_dict["train_lexical"], X_meta], format="csr")

        elif branch_name == "community_bagger":
            # Sparse Community + Dense Meta
            return sp.hstack([features_dict["train_community"], X_meta], format="csr")

        elif branch_name in [
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
        ]:
            # Dense Semantic + Dense Meta
            return np.hstack([features_dict["train_semantic"], X_meta])

        elif branch_name in ["metadata_anchor", "temporal_booster"]:
            # Pure Meta
            return X_meta

        else:
            raise ValueError(f"Unknown branch: {branch_name}")

    def _train_volatile(self, model_name, model, X, y, skf):
        """
        Trains a Volatile Learner using CV-Bagging with Early Stopping.
        Saves a model for every fold.
        """
        oof_preds = np.zeros(len(y))
        fold_scores = []

        logger.info(f"Starting Volatile Training for {model_name}...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Clone model to ensure fresh start
            clf = clone(model)

            # Fit with Early Stopping
            # Note: Early stopping parameters are handled differently for XGB/LGBM sklearn APIs
            fit_params = {}
            if isinstance(clf, xgb.XGBClassifier):
                fit_params = {"eval_set": [(X_val_fold, y_val_fold)], "verbose": False}
            elif isinstance(clf, lgb.LGBMClassifier):
                # LightGBM sklearn API
                fit_params = {
                    "eval_set": [(X_val_fold, y_val_fold)],
                    "eval_metric": "auc",
                    "callbacks": [
                        lgb.early_stopping(
                            stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                        ),
                        lgb.log_evaluation(period=0),  # Silence
                    ],
                }

            clf.fit(X_train_fold, y_train_fold, **fit_params)

            # Predict (Use best iteration automatically for XGB/LGBM if early stopping was used)
            val_probs = clf.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = val_probs

            score = roc_auc_score(y_val_fold, val_probs)
            fold_scores.append(score)
            logger.info(f"  {model_name} Fold {fold} AUC: {score:.16f}")

            # Save Fold Model
            save_path = os.path.join(self.model_dir, f"{model_name}_fold_{fold}.joblib")
            joblib.dump(clf, save_path)

        avg_score = np.mean(fold_scores)
        logger.info(f"{model_name} CV Average AUC: {avg_score:.16f}")
        return oof_preds

    def _train_stable(self, model_name, model, X, y, skf):
        """
        Trains a Stable Learner.
        1. Performs CV to generate OOF predictions (saving fold models as artifacts).
        2. Retrains a single model on the FULL Union Dataset for inference.
        """
        oof_preds = np.zeros(len(y))
        fold_scores = []

        logger.info(f"Starting Stable Training for {model_name}...")

        # Step 1: OOF Generation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold = X[val_idx]

            clf = clone(model)
            clf.fit(X_train_fold, y_train_fold)

            val_probs = clf.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = val_probs

            score = roc_auc_score(y[val_idx], val_probs)
            fold_scores.append(score)

            # Save Fold Model (as per requirement "Save all 7 models for this fold")
            save_path = os.path.join(self.model_dir, f"{model_name}_fold_{fold}.joblib")
            joblib.dump(clf, save_path)

        avg_score = np.mean(fold_scores)
        logger.info(f"{model_name} CV Average AUC (OOF): {avg_score:.16f}")

        # Step 2: Full Retrain
        logger.info(f"Retraining {model_name} on full Union Dataset...")
        full_model = clone(model)
        full_model.fit(X, y)

        save_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(full_model, save_path)
        logger.info(f"Saved full {model_name} to {save_path}")

        return oof_preds

    def train_level_1_ensemble(self, features_dict):
        """
        Iterates through all 7 branches, training them according to their type.
        Returns a DataFrame of OOF predictions.
        """
        y = features_dict["y_train"]
        oof_df = pd.DataFrame()

        # Calculate scale_pos_weight for XGBoost
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        for name, config in self.branches.items():
            logger.info(f"\n{'='*20} Branch: {name} {'='*20}")

            # 1. Prepare Data
            X = self._prepare_features(features_dict, name)

            # 2. Instantiate Model
            factory_kwargs = {}
            if name == "semantic_booster":
                factory_kwargs["scale_pos_weight"] = scale_pos_weight

            model = config["factory"](**factory_kwargs)

            # 3. Train
            if config["type"] == "volatile":
                oof = self._train_volatile(name, model, X, y, skf)
            else:
                oof = self._train_stable(name, model, X, y, skf)

            oof_df[name] = oof

        # Save OOFs
        oof_df["target"] = y
        oof_df.to_csv(self.oof_path, index=False)
        logger.info(f"Saved Level 1 OOF predictions to {self.oof_path}")

        return oof_df

    def train_level_2_meta(self, oof_df):
        """
        Trains the Level 2 Logistic Regression Meta-Learner on the OOF predictions.
        """
        logger.info(f"\n{'='*20} Level 2: Meta Learner {'='*20}")

        feature_cols = [c for c in oof_df.columns if c != "target"]
        X = oof_df[feature_cols].values
        y = oof_df["target"].values

        meta_learner = ModelFactory.get_meta_learner()

        # We can use CV here to estimate meta-performance, but we train on full OOFs for final model
        scores = []
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = clone(meta_learner)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            score = roc_auc_score(y[val_idx], probs)
            scores.append(score)
            logger.info(f"  Meta-Learner Fold {fold} AUC: {score:.16f}")

        logger.info(f"Meta-Learner CV Average AUC: {np.mean(scores):.16f}")

        # Final Train
        meta_learner.fit(X, y)
        save_path = os.path.join(self.model_dir, "meta_learner.joblib")
        joblib.dump(meta_learner, save_path)
        logger.info(f"Saved Meta-Learner to {save_path}")

    def run(self, load_cached_data=True, debug_size=None):
        """
        Main execution method.
        """
        set_seed(self.random_state)

        # 1. Load Union Dataset
        train_df, test_df = load_union_dataset(
            load_cached_data=load_cached_data, debug_size=debug_size
        )

        # 2. Feature Engineering
        fe_pipeline = FeatureEngineeringPipeline()
        features_dict = fe_pipeline.run(
            train_df, test_df, load_cached_data=load_cached_data
        )

        # 3. Train Level 1 Ensemble
        oof_df = self.train_level_1_ensemble(features_dict)

        # 4. Train Level 2 Meta Learner
        self.train_level_2_meta(oof_df)

        logger.info("Training Pipeline Completed Successfully.")
