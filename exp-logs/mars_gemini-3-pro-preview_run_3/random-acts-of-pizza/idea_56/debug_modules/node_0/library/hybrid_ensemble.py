import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, load_data, get_score, save_submission
from library.features import FeaturePipeline
from library.model_definitions import get_model_registry, get_meta_learner


class HybridStackingEnsemble:
    """
    Deca-View Full-Spectrum Stacking Ensemble with Hybrid Inference Protocol.
    """

    def __init__(self):
        set_seed()
        self.registry = get_model_registry()
        self.meta_learner = get_meta_learner()
        self.feature_pipeline = FeaturePipeline()

        # Storage for trained models
        # Structure: {'model_name': {'folds': [model_fold_0, ...], 'full': model_full}}
        self.trained_models = {
            k: {"folds": [], "full": None} for k in self.registry.keys()
        }

        # Storage for OOF predictions (Level 1 output)
        self.oof_preds = None
        self.feature_cache = {}

    def _concat_features(self, feature_dict, keys):
        """
        Concatenates selected feature sets (sparse or dense) into a single matrix.
        """
        selected = [feature_dict[k] for k in keys]

        # Check if any input is sparse
        is_sparse = any(sp.issparse(x) for x in selected)

        if is_sparse:
            # Convert dense arrays to sparse matrices for stacking
            processed = []
            for x in selected:
                if sp.issparse(x):
                    processed.append(x)
                else:
                    processed.append(sp.csr_matrix(x))
            return sp.hstack(processed, format="csr")
        else:
            return np.hstack(selected)

    def fit(self, load_cached_data=True):
        """
        Executes the training pipeline:
        1. Load Data
        2. Generate Features
        3. Level 1: 5-Fold CV (Volatile w/ ES, Stable w/o ES)
        4. Hybrid Retraining (Stable on Full Train+Val)
        5. Level 2: Train Meta-Learner on OOF
        """
        print("Loading data...")
        train_df, val_df, test_df = load_data()

        # Target extraction
        y_train = train_df[Config.TARGET_COL].values
        y_val = val_df[Config.TARGET_COL].values

        # Combined Full Train (Train + Val) for Stable Retraining
        full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        y_full = full_train_df[Config.TARGET_COL].values

        print("Generating features...")
        # Fit pipeline on Train, transform others
        feats_train = self.feature_pipeline.fit_transform(
            train_df, split_name="train", load_cached_data=load_cached_data
        )
        feats_val = self.feature_pipeline.transform(
            val_df, split_name="val", load_cached_data=load_cached_data
        )

        # For full retraining, we need features for the combined dataset.
        # We can efficiently stack them since pipeline was fit on train and applied to val.
        feats_full = {}
        for key in feats_train.keys():
            if sp.issparse(feats_train[key]):
                feats_full[key] = sp.vstack(
                    [feats_train[key], feats_val[key]], format="csr"
                )
            else:
                feats_full[key] = np.vstack([feats_train[key], feats_val[key]])

        # Initialize OOF matrix: (n_samples, n_models)
        self.oof_preds = pd.DataFrame(
            np.zeros((len(train_df), len(self.registry))),
            columns=self.registry.keys(),
            index=train_df.index,
        )

        # --- Level 1: 5-Fold Stratified CV ---
        print(f"Starting Level 1 Training ({Config.N_FOLDS}-Fold CV)...")
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        for fold, (train_idx, cv_val_idx) in enumerate(skf.split(train_df, y_train)):
            print(f"  Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            # Get Fold Targets
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[cv_val_idx]

            for name, config in self.registry.items():
                estimator = clone(config["estimator"])
                feature_keys = config["feature_sets"]
                is_volatile = config["is_volatile"]

                # Prepare Fold Data
                X_fold_train = self._concat_features(
                    {k: v[train_idx] for k, v in feats_train.items()}, feature_keys
                )
                X_fold_val = self._concat_features(
                    {k: v[cv_val_idx] for k, v in feats_train.items()}, feature_keys
                )

                # Train
                if is_volatile:
                    # Volatile models (XGB/LGBM) use Early Stopping
                    # Note: early_stopping_rounds is passed via config params to init,
                    # but sklearn API usually requires eval_set in fit.
                    estimator.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False,
                    )
                    # Persist fold model
                    self.trained_models[name]["folds"].append(estimator)
                else:
                    # Stable models just fit
                    estimator.fit(X_fold_train, y_fold_train)
                    # We do NOT persist fold models for stable learners (we retrain full later),
                    # but we need them now for OOF predictions.

                # Predict OOF
                if hasattr(estimator, "predict_proba"):
                    preds = estimator.predict_proba(X_fold_val)[:, 1]
                else:
                    preds = estimator.predict(X_fold_val)

                self.oof_preds.iloc[
                    cv_val_idx, self.oof_preds.columns.get_loc(name)
                ] = preds

        # Print OOF Scores
        print("\nLevel 1 OOF Scores (ROC-AUC):")
        for name in self.registry.keys():
            score = roc_auc_score(y_train, self.oof_preds[name])
            print(f"  {name}: {score}")

        # --- Hybrid Retraining ---
        print("\nStarting Hybrid Retraining...")
        for name, config in self.registry.items():
            if not config["is_volatile"]:
                print(f"  Retraining Stable Model: {name} on Full Train + Val...")
                estimator = clone(config["estimator"])
                feature_keys = config["feature_sets"]

                # Prepare Full Data
                X_full_combined = self._concat_features(feats_full, feature_keys)

                estimator.fit(X_full_combined, y_full)
                self.trained_models[name]["full"] = estimator
            else:
                print(
                    f"  Skipping Retraining for Volatile Model: {name} (Using {len(self.trained_models[name]['folds'])} fold models)"
                )

        # --- Level 2: Meta-Learner Training ---
        print("\nTraining Level 2 Meta-Learner...")
        self.meta_learner.fit(self.oof_preds, y_train)

        # Evaluate Meta-Learner on OOF
        meta_oof_preds = self.meta_learner.predict_proba(self.oof_preds)[:, 1]
        score = roc_auc_score(y_train, meta_oof_preds)
        print(f"Meta-Learner OOF ROC-AUC: {score}")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set.
        1. Load Test Data
        2. Generate Features
        3. Level 1 Inference (Hybrid: Full model for Stable, CV-Bagging for Volatile)
        4. Level 2 Inference
        5. Save Submission
        """
        print("\nStarting Inference...")
        _, _, test_df = load_data()

        print("Generating Test features...")
        feats_test = self.feature_pipeline.transform(
            test_df, split_name="test", load_cached_data=load_cached_data
        )

        level1_test_preds = pd.DataFrame(
            index=test_df.index, columns=self.registry.keys()
        )

        print("Generating Level 1 Predictions...")
        for name, config in self.registry.items():
            feature_keys = config["feature_sets"]
            X_test = self._concat_features(feats_test, feature_keys)

            if config["is_volatile"]:
                # Volatile: Average prediction of all fold models
                fold_preds = []
                for model in self.trained_models[name]["folds"]:
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(X_test)[:, 1]
                    else:
                        p = model.predict(X_test)
                    fold_preds.append(p)
                level1_test_preds[name] = np.mean(fold_preds, axis=0)
            else:
                # Stable: Use single fully retrained model
                model = self.trained_models[name]["full"]
                if hasattr(model, "predict_proba"):
                    level1_test_preds[name] = model.predict_proba(X_test)[:, 1]
                else:
                    level1_test_preds[name] = model.predict(X_test)

        print("Generating Level 2 Predictions...")
        final_preds = self.meta_learner.predict_proba(level1_test_preds)[:, 1]

        print("Saving Submission...")
        save_submission(test_df[Config.ID_COL].values, final_preds)
        return final_preds
