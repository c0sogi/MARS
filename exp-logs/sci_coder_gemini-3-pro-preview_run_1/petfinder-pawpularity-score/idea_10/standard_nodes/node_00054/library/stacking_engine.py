import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.feature_processor import FeatureProcessor


class Level0Trainer:
    """
    Trains Level-0 experts (Ridge, SVR, ExtraTrees) using 5-Fold CV.
    Manages feature processing and caching of OOF/Test predictions.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self, backbone_name, model_type, suffix=""):
        base_name = f"{backbone_name}_{model_type}{suffix}"
        return {
            "oof": os.path.join(self.cache_dir, f"{base_name}_oof.npy"),
            "test": os.path.join(self.cache_dir, f"{base_name}_test_pred.npy"),
        }

    def _train_ridge(self, X_train, y_train, X_val, X_test):
        # RidgeCV with built-in LOO-CV for alpha selection
        model = RidgeCV(
            alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error"
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        return val_pred, test_pred

    def _train_svr(self, X_train, y_train, X_val, X_test):
        # SVR with GridSearchCV
        # Using a smaller inner CV (e.g., 3) to keep runtime manageable
        estimator = SVR(kernel="rbf")
        model = GridSearchCV(
            estimator,
            Config.SVR_GRID,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        return val_pred, test_pred

    def _train_et(self, X_train, y_train, X_val, X_test):
        # ExtraTrees with GridSearchCV
        estimator = ExtraTreesRegressor(**Config.ET_PARAMS)
        model = GridSearchCV(
            estimator,
            Config.ET_GRID,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        return val_pred, test_pred

    def run_expert(
        self,
        backbone_name,
        model_type,
        train_data,
        test_data,
        load_cached_data=True,
        cache_suffix="",
    ):
        """
        Runs the training pipeline for a specific backbone and model type.

        Args:
            backbone_name (str): Name of the backbone (e.g., 'siglip').
            model_type (str): Type of model ('ridge', 'svr', 'et').
            train_data (dict): Dictionary containing 'features', 'meta', 'targets'.
            test_data (dict): Dictionary containing 'features', 'meta'.
            load_cached_data (bool): Whether to load from cache if available.
            cache_suffix (str): Suffix to append to cache filenames (e.g., "_full").

        Returns:
            tuple: (oof_predictions, test_predictions)
        """
        paths = self._get_cache_paths(backbone_name, model_type, suffix=cache_suffix)

        # 1. Check Cache
        if load_cached_data:
            if os.path.exists(paths["oof"]) and os.path.exists(paths["test"]):
                print(f"[{backbone_name} - {model_type}] Loading cached predictions...")
                try:
                    cached_oof = np.load(paths["oof"])
                    cached_test = np.load(paths["test"])

                    # Validate dimensions against current runtime configuration (Cite debug_lesson_5)
                    if len(cached_oof) == len(train_data["targets"]) and len(
                        cached_test
                    ) == len(test_data["features"]):
                        return cached_oof, cached_test
                    else:
                        print(
                            f"Cache dimension mismatch (OOF: {len(cached_oof)} vs {len(train_data['targets'])}). Recomputing..."
                        )
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing...")

        print(f"[{backbone_name} - {model_type}] Training expert...")

        # 2. Prepare Data
        train_features = train_data["features"]
        train_meta = train_data["meta"]
        train_targets = train_data["targets"]

        test_features = test_data["features"]
        test_meta = test_data["meta"]

        oof_preds = np.zeros(len(train_targets), dtype=np.float32)
        test_preds_folds = []

        # 3. Cross-Validation Loop
        for fold, (train_idx, val_idx) in enumerate(
            self.kf.split(train_features, train_targets)
        ):
            # Split Data
            X_feat_tr, X_feat_val = train_features[train_idx], train_features[val_idx]
            X_meta_tr, X_meta_val = train_meta[train_idx], train_meta[val_idx]
            y_tr, y_val = train_targets[train_idx], train_targets[val_idx]

            # Initialize Feature Processor for this fold
            processor = FeatureProcessor()

            # Transform Features based on model type
            if model_type in ["ridge", "svr"]:
                # Linear models: Scale everything
                X_tr_proc = processor.prepare_data_for_linear(
                    X_feat_tr, X_meta_tr, fit=True
                )
                X_val_proc = processor.prepare_data_for_linear(
                    X_feat_val, X_meta_val, fit=False
                )
                X_test_proc = processor.prepare_data_for_linear(
                    test_features, test_meta, fit=False
                )

                if model_type == "ridge":
                    val_p, test_p = self._train_ridge(
                        X_tr_proc, y_tr, X_val_proc, X_test_proc
                    )
                else:
                    val_p, test_p = self._train_svr(
                        X_tr_proc, y_tr, X_val_proc, X_test_proc
                    )

            elif model_type == "et":
                # Tree models: PCA on embeddings + Raw Metadata
                X_tr_proc = processor.prepare_data_for_tree(
                    X_feat_tr, X_meta_tr, fit=True
                )
                X_val_proc = processor.prepare_data_for_tree(
                    X_feat_val, X_meta_val, fit=False
                )
                X_test_proc = processor.prepare_data_for_tree(
                    test_features, test_meta, fit=False
                )

                val_p, test_p = self._train_et(X_tr_proc, y_tr, X_val_proc, X_test_proc)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Store predictions
            oof_preds[val_idx] = val_p
            test_preds_folds.append(test_p)

            fold_rmse = compute_rmse(y_val, val_p)
            print(f"  Fold {fold + 1} RMSE: {fold_rmse}")

        # 4. Aggregate Test Predictions
        avg_test_preds = np.mean(test_preds_folds, axis=0)

        # 5. Report Overall RMSE
        total_rmse = compute_rmse(train_targets, oof_preds)
        print(f"[{backbone_name} - {model_type}] Overall OOF RMSE: {total_rmse}")

        # 6. Save to Cache
        np.save(paths["oof"], oof_preds)
        np.save(paths["test"], avg_test_preds)

        return oof_preds, avg_test_preds


class Level1MetaLearner:
    """
    Aggregates Level-0 predictions using Bayesian Ridge Regression.
    Performs Nested CV for evaluation and generates final submission.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    def train_and_predict(
        self, oof_dict, test_pred_dict, y_true, test_ids, return_oof=False
    ):
        """
        Args:
            oof_dict (dict): Map of 'expert_name' -> oof_predictions array.
            test_pred_dict (dict): Map of 'expert_name' -> test_predictions array.
            y_true (array): Ground truth targets for training set.
            test_ids (array): IDs for the test set (for submission file).
            return_oof (bool): If True, returns (test_preds, oof_preds) instead of just test_preds.
        """
        # Sort keys to ensure consistent column order
        feature_names = sorted(oof_dict.keys())
        print(f"\n[Meta-Learner] Aggregating features from: {feature_names}")

        # Stack features
        X_meta_train = np.column_stack([oof_dict[k] for k in feature_names])
        X_meta_test = np.column_stack([test_pred_dict[k] for k in feature_names])

        # 1. Nested CV Evaluation
        print("[Meta-Learner] Running Nested CV Evaluation...")
        meta_oof_preds = np.zeros(len(y_true))

        for fold, (train_idx, val_idx) in enumerate(
            self.kf.split(X_meta_train, y_true)
        ):
            X_tr, X_val = X_meta_train[train_idx], X_meta_train[val_idx]
            y_tr, y_val = y_true[train_idx], y_true[val_idx]

            # Cite debug_lesson_3: Handle potential stale config (n_iter) in persistent envs
            params = Config.META_MODEL_PARAMS.copy()
            if "n_iter" in params:
                params["max_iter"] = params.pop("n_iter")

            model = BayesianRidge(**params)
            model.fit(X_tr, y_tr)
            meta_oof_preds[val_idx] = model.predict(X_val)

        cv_score = compute_rmse(y_true, meta_oof_preds)
        print(f"[Meta-Learner] Nested CV RMSE: {cv_score}")

        # 2. Final Training and Prediction
        print("[Meta-Learner] Training final model on full OOF data...")

        # Cite debug_lesson_3: Handle potential stale config (n_iter) in persistent envs
        params = Config.META_MODEL_PARAMS.copy()
        if "n_iter" in params:
            params["max_iter"] = params.pop("n_iter")

        final_model = BayesianRidge(**params)
        final_model.fit(X_meta_train, y_true)

        final_test_preds = final_model.predict(X_meta_test)

        # 3. Generate Submission
        self._save_submission(test_ids, final_test_preds)

        if return_oof:
            return final_test_preds, meta_oof_preds
        return final_test_preds

    def _save_submission(self, ids, preds):
        submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: preds})

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
