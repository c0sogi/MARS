import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import (
    load_array,
    save_array,
    check_cache_exists,
    set_seed,
    rmse_score,
)


class Level0Experts:
    """
    Implements the Level-0 Experts training and inference pipeline.
    Trains 12 experts (3 Backbones x 4 Algorithms) using Stratified 5-Fold CV.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.models = ["ridge", "svr", "et", "lgbm"]
        self.backbones = list(Config.BACKBONES.keys())

    def _load_and_merge_data(self, backbone_key):
        """
        Loads features, ids, meta, and targets for train/val/test splits
        and merges train+val for cross-validation.
        """
        # Load Train
        train_feats = load_array(f"{backbone_key}_train_features.npy")
        train_ids = load_array(f"{backbone_key}_train_ids.npy")
        train_meta = load_array(f"{backbone_key}_train_meta.npy")
        train_targets = load_array(f"{backbone_key}_train_targets.npy")

        # Load Val
        val_feats = load_array(f"{backbone_key}_val_features.npy")
        val_ids = load_array(f"{backbone_key}_val_ids.npy")
        val_meta = load_array(f"{backbone_key}_val_meta.npy")
        val_targets = load_array(f"{backbone_key}_val_targets.npy")

        # Load Test
        test_feats = load_array(f"{backbone_key}_test_features.npy")
        test_ids = load_array(f"{backbone_key}_test_ids.npy")
        test_meta = load_array(f"{backbone_key}_test_meta.npy")
        # Test targets are usually dummy values, but we load them for consistency if needed
        # test_targets = load_array(f"{backbone_key}_test_targets.npy")

        # Merge Train and Val
        X_dev = np.concatenate([train_feats, val_feats], axis=0)
        ids_dev = np.concatenate([train_ids, val_ids], axis=0)
        meta_dev = np.concatenate([train_meta, val_meta], axis=0)
        y_dev = np.concatenate([train_targets, val_targets], axis=0)

        if self.debug:
            limit = 200
            X_dev = X_dev[:limit]
            ids_dev = ids_dev[:limit]
            meta_dev = meta_dev[:limit]
            y_dev = y_dev[:limit]
            test_feats = test_feats[:20]
            test_ids = test_ids[:20]
            test_meta = test_meta[:20]

        return {
            "X_dev": X_dev,
            "ids_dev": ids_dev,
            "meta_dev": meta_dev,
            "y_dev": y_dev,
            "X_test": test_feats,
            "ids_test": test_ids,
            "meta_test": test_meta,
        }

    def _get_stratified_folds(self, y, n_folds):
        """
        Creates stratified folds based on binned continuous target.
        """
        # Bin targets for stratification (Sturges' rule approx or fixed bins)
        num_bins = int(np.floor(1 + np.log2(len(y))))
        y_binned = pd.cut(y, bins=num_bins, labels=False)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
        return list(skf.split(np.zeros(len(y)), y_binned))

    def _preprocess_fold(
        self, model_type, X_train, meta_train, X_val, meta_val, X_test, meta_test
    ):
        """
        Applies specific preprocessing based on model type.
        """
        if model_type in ["ridge", "svr"]:
            # Strategy: Concatenate [Image, Meta] -> StandardScaler
            # Note: Meta is binary, but scaling it with image features is specified in strategy
            # to treat the full vector uniformly for distance/linear based models.

            # Concatenate first
            train_full = np.hstack([X_train, meta_train])
            val_full = np.hstack([X_val, meta_val])
            test_full = np.hstack([X_test, meta_test])

            scaler = StandardScaler()
            train_out = scaler.fit_transform(train_full)
            val_out = scaler.transform(val_full)
            test_out = scaler.transform(test_full)

            return train_out, val_out, test_out

        elif model_type in ["et", "lgbm"]:
            # Strategy: PCA on Image -> Concatenate [PCA_Image, Meta]
            pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)

            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)
            X_test_pca = pca.transform(X_test)

            train_out = np.hstack([X_train_pca, meta_train])
            val_out = np.hstack([X_val_pca, meta_val])
            test_out = np.hstack([X_test_pca, meta_test])

            return train_out, val_out, test_out

        return X_train, X_val, X_test

    def _train_predict_model(self, model_type, X_train, y_train, X_val, y_val, X_test):
        """
        Trains a specific model type and returns predictions.
        """
        model = None

        if model_type == "ridge":
            model = RidgeCV(
                alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error"
            )
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        elif model_type == "svr":
            model = SVR(**Config.SVR_PARAMS)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        elif model_type == "et":
            model = ExtraTreesRegressor(**Config.ET_PARAMS)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        elif model_type == "lgbm":
            # LightGBM with early stopping
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=Config.LGBM_ES_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Silence
            ]

            model = lgb.train(
                Config.LGBM_PARAMS,
                train_data,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        return val_pred, test_pred

    def run(self, load_cached_data: bool = True):
        """
        Main execution method.
        """
        set_seed(Config.SEED)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        print("Starting Level-0 Experts Training...")

        for backbone in self.backbones:
            print(f"\n=== Processing Backbone: {backbone} ===")

            # Load data once per backbone
            data = self._load_and_merge_data(backbone)
            X_dev, y_dev = data["X_dev"], data["y_dev"]
            meta_dev = data["meta_dev"]
            ids_dev = data["ids_dev"]

            X_test_raw = data["X_test"]
            meta_test = data["meta_test"]
            ids_test = data["ids_test"]

            # Save the merged dev targets/IDs/Meta once per backbone (or globally)
            # These are needed to align OOF predictions in Level 1
            save_array(f"{backbone}_merged_ids.npy", ids_dev)
            save_array(f"{backbone}_merged_targets.npy", y_dev)
            save_array(f"{backbone}_merged_meta.npy", meta_dev)
            save_array(f"{backbone}_test_ids_ref.npy", ids_test)
            save_array(f"{backbone}_test_meta_ref.npy", meta_test)

            # Generate folds
            folds = self._get_stratified_folds(y_dev, Config.N_FOLDS)

            for model_name in self.models:
                # Check cache
                oof_file = f"{backbone}_{model_name}_oof.npy"
                test_pred_file = f"{backbone}_{model_name}_test_pred.npy"

                if (
                    load_cached_data
                    and check_cache_exists(oof_file)
                    and check_cache_exists(test_pred_file)
                ):
                    print(f"Skipping {model_name} (Cached)")
                    continue

                print(f"Training {model_name}...")

                oof_preds = np.zeros(len(y_dev))
                test_preds_accum = np.zeros((Config.N_FOLDS, len(X_test_raw)))

                fold_scores = []

                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    # Split Data
                    X_train, X_val = X_dev[train_idx], X_dev[val_idx]
                    y_train, y_val = y_dev[train_idx], y_dev[val_idx]
                    meta_train, meta_val = meta_dev[train_idx], meta_dev[val_idx]

                    # Preprocess
                    X_train_proc, X_val_proc, X_test_proc = self._preprocess_fold(
                        model_name,
                        X_train,
                        meta_train,
                        X_val,
                        meta_val,
                        X_test_raw,
                        meta_test,
                    )

                    # Train & Predict
                    val_pred, test_pred = self._train_predict_model(
                        model_name,
                        X_train_proc,
                        y_train,
                        X_val_proc,
                        y_val,
                        X_test_proc,
                    )

                    # Store
                    oof_preds[val_idx] = val_pred
                    test_preds_accum[fold_idx] = test_pred

                    score = rmse_score(y_val, val_pred)
                    fold_scores.append(score)

                # Average Test Predictions
                avg_test_preds = np.mean(test_preds_accum, axis=0)

                # Report
                overall_rmse = rmse_score(y_dev, oof_preds)
                print(f"  {model_name.upper()} OOF RMSE: {overall_rmse}")

                # Save
                save_array(oof_file, oof_preds)
                save_array(test_pred_file, avg_test_preds)

        print("\nLevel-0 Training Complete.")


def train_level0_experts(debug: bool = False, load_cached_data: bool = True):
    """
    Public interface to run the Level-0 training pipeline.
    """
    pipeline = Level0Experts(debug=debug)
    pipeline.run(load_cached_data=load_cached_data)
