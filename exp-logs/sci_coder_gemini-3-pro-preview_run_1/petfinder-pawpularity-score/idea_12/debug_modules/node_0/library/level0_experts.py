import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.feature_processor import ExpertPreprocessor


class Level0Trainer:
    """
    Manages the training and prediction of Level-0 heterogeneous experts.
    Implements Stratified K-Fold CV, model-specific preprocessing, and caching.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED
        seed_everything(self.seed)

    def _get_cache_paths(self, backbone_name, expert_name):
        """Generates file paths for caching OOF and Test predictions."""
        base_name = f"{backbone_name}_{expert_name}"
        oof_path = os.path.join(self.working_dir, f"{base_name}_oof.npy")
        test_pred_path = os.path.join(self.working_dir, f"{base_name}_test_pred.npy")
        return oof_path, test_pred_path

    def _create_stratified_folds(self, targets):
        """
        Creates Stratified K-Fold iterators by binning the continuous target.
        """
        # Sturges' rule for binning: k = 1 + 3.322 log_10(N)
        # For N ~ 9000, k ~ 14
        num_bins = 14
        # Handle edge case where targets might be constant or too few
        if len(np.unique(targets)) < num_bins:
            bins = targets
        else:
            bins = pd.cut(targets, bins=num_bins, labels=False)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        return list(skf.split(np.zeros(len(targets)), bins))

    def _train_ridge(self, X_train, y_train, X_val, y_val):
        """Trains RidgeCV."""
        # RidgeCV performs efficient LOO-CV internally to pick alpha
        model = RidgeCV(
            alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error"
        )
        model.fit(X_train, y_train)
        return model

    def _train_svr(self, X_train, y_train, X_val, y_val):
        """Trains SVR with GridSearchCV."""
        # GridSearch for C and epsilon
        # Using n_jobs=-1 to speed up search
        param_grid = Config.SVR_PARAMS
        model = GridSearchCV(
            SVR(),
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,  # Internal CV for hyperparam tuning
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.best_estimator_

    def _train_extratrees(self, X_train, y_train, X_val, y_val):
        """Trains ExtraTreesRegressor."""
        model = ExtraTreesRegressor(**Config.ET_PARAMS)
        model.fit(X_train, y_train)
        return model

    def _train_lgbm(self, X_train, y_train, X_val, y_val):
        """Trains LGBMRegressor with Early Stopping."""
        model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=0),  # Suppress logging
        ]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        return model

    def train_expert(
        self,
        backbone_name: str,
        expert_name: str,
        train_embeddings: np.ndarray,
        train_metadata: np.ndarray,
        train_targets: np.ndarray,
        test_embeddings: np.ndarray,
        test_metadata: np.ndarray,
        load_cached_data: bool = True,
    ):
        """
        Trains a specific expert (algorithm) on features from a specific backbone.

        Args:
            backbone_name: Name of the backbone (e.g., 'siglip').
            expert_name: Name of the algorithm ('ridge', 'svr', 'et', 'lgbm').
            train_embeddings: Merged Train+Val embeddings.
            train_metadata: Merged Train+Val metadata.
            train_targets: Merged Train+Val targets.
            test_embeddings: Test embeddings.
            test_metadata: Test metadata.
            load_cached_data: Whether to load predictions from disk if available.

        Returns:
            oof_preds: Out-of-fold predictions for training data.
            test_preds: Averaged predictions for test data.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        oof_path, test_pred_path = self._get_cache_paths(backbone_name, expert_name)

        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(oof_path)
            and os.path.exists(test_pred_path)
        ):
            # print(f"Loading cached Level-0 predictions for {backbone_name} - {expert_name}")
            return np.load(oof_path), np.load(test_pred_path)

        # print(f"Training Level-0 Expert: {backbone_name} - {expert_name}")

        # 2. Setup
        n_train = len(train_targets)
        n_test = len(test_embeddings)

        oof_preds = np.zeros(n_train)
        test_preds_accum = np.zeros(n_test)

        folds = self._create_stratified_folds(train_targets)

        # Determine Preprocessing Strategy
        is_tree = expert_name in ["et", "lgbm"]

        # 3. Cross-Validation Loop
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Split Data
            X_emb_tr, X_emb_val = train_embeddings[train_idx], train_embeddings[val_idx]
            X_meta_tr, X_meta_val = train_metadata[train_idx], train_metadata[val_idx]
            y_tr, y_val = train_targets[train_idx], train_targets[val_idx]

            # Preprocessing
            preprocessor = ExpertPreprocessor(
                pca_components=Config.PCA_COMPONENTS, random_state=self.seed
            )

            if is_tree:
                # PCA on embeddings + Raw Metadata
                preprocessor.fit_tree(X_emb_tr)
                X_tr_fold = preprocessor.transform_tree(X_emb_tr, X_meta_tr)
                X_val_fold = preprocessor.transform_tree(X_emb_val, X_meta_val)
                X_test_fold = preprocessor.transform_tree(
                    test_embeddings, test_metadata
                )
            else:
                # Concat + StandardScaler
                preprocessor.fit_linear(X_emb_tr, X_meta_tr)
                X_tr_fold = preprocessor.transform_linear(X_emb_tr, X_meta_tr)
                X_val_fold = preprocessor.transform_linear(X_emb_val, X_meta_val)
                X_test_fold = preprocessor.transform_linear(
                    test_embeddings, test_metadata
                )

            # Model Training
            if expert_name == "ridge":
                model = self._train_ridge(X_tr_fold, y_tr, X_val_fold, y_val)
            elif expert_name == "svr":
                model = self._train_svr(X_tr_fold, y_tr, X_val_fold, y_val)
            elif expert_name == "et":
                model = self._train_extratrees(X_tr_fold, y_tr, X_val_fold, y_val)
            elif expert_name == "lgbm":
                model = self._train_lgbm(X_tr_fold, y_tr, X_val_fold, y_val)
            else:
                raise ValueError(f"Unknown expert name: {expert_name}")

            # Prediction
            val_pred = model.predict(X_val_fold)
            test_pred = model.predict(X_test_fold)

            # Enforce bounds (Pawpularity is 1-100)
            val_pred = np.clip(val_pred, 1.0, 100.0)
            test_pred = np.clip(test_pred, 1.0, 100.0)

            oof_preds[val_idx] = val_pred
            test_preds_accum += test_pred

            # Optional: Print fold metric
            # fold_rmse = compute_rmse(y_val, val_pred)
            # print(f"  Fold {fold_idx+1} RMSE: {fold_rmse:.5f}")

        # 4. Finalize
        avg_test_preds = test_preds_accum / self.n_folds

        full_rmse = compute_rmse(train_targets, oof_preds)
        print(f"Expert {backbone_name}-{expert_name} CV RMSE: {full_rmse:.10f}")

        # 5. Cache
        np.save(oof_path, oof_preds)
        np.save(test_pred_path, avg_test_preds)

        return oof_preds, avg_test_preds
