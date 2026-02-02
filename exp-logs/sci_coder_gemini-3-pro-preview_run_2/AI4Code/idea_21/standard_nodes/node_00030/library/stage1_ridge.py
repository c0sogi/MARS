import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config


class RidgeStacker:
    """
    Implements the Stage 1 Sparse Lexical Regressor using Ridge Regression.

    Responsibilities:
    1. Generate Out-Of-Fold (OOF) predictions via GroupKFold CV (for Stage 2 features).
    2. Train a final model on the full dataset (for Test inference).
    3. Manage caching of OOF predictions and the trained model.
    """

    def __init__(self):
        self.config = Config
        self.model_params = self.config.RIDGE_PARAMS
        self.n_folds = self.config.N_FOLDS_STAGE1
        self.working_dir = self.config.WORKING_DIR

        # Initialize the model
        self.model = Ridge(**self.model_params)
        self.is_fitted = False

    def _get_paths(self):
        """Returns paths for caching OOF predictions and the model."""
        oof_path = os.path.join(self.working_dir, "stage1_oof_preds.npy")
        model_path = os.path.join(self.working_dir, "stage1_ridge_model.joblib")
        return oof_path, model_path

    def fit_predict_oof(self, X, y, groups, load_cached_data=True):
        """
        Performs Group K-Fold Cross Validation to generate OOF predictions,
        then refits the model on the full dataset.

        Args:
            X (scipy.sparse.csr_matrix): Feature matrix (TF-IDF).
            y (np.array): Target values (normalized ranks).
            groups (np.array): Group labels for CV (ancestor_ids).
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            np.array: Out-Of-Fold predictions for the training set.
        """
        oof_path, model_path = self._get_paths()

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(oof_path) and os.path.exists(model_path):
                print(f"Loading Stage 1 OOF predictions from {oof_path}")
                print(f"Loading Stage 1 Model from {model_path}")
                try:
                    oof_preds = np.load(oof_path)
                    self.model = joblib.load(model_path)
                    self.is_fitted = True
                    return oof_preds
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing...")
            else:
                print("Stage 1 cache not found. Proceeding to fit...")
        else:
            print("Stage 1 cache loading disabled. Proceeding to fit...")

        # 2. Setup Cross-Validation
        gkf = GroupKFold(n_splits=self.n_folds)
        oof_preds = np.zeros(X.shape[0])

        print(f"Starting Stage 1 Ridge CV ({self.n_folds} folds)...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            # Train fold model
            fold_model = Ridge(**self.model_params)
            fold_model.fit(X_train, y_train)

            # Predict
            val_preds = fold_model.predict(X_val)

            # Clip predictions to valid range [0, 1] for sanity,
            # though Ridge can extrapolate.
            # We keep raw outputs for the stacker usually, but clipping
            # helps interpretation of MAE.
            oof_preds[val_idx] = val_preds

            # Optional: Print fold metric
            # fold_mae = mean_absolute_error(y[val_idx], val_preds)
            # print(f"Fold {fold+1} MAE: {fold_mae}")

        # 3. Report Metrics
        overall_mae = mean_absolute_error(y, oof_preds)
        print(f"Stage 1 OOF MAE: {overall_mae}")

        # 4. Refit on Full Data
        print("Retraining Stage 1 Ridge on full dataset...")
        self.model.fit(X, y)
        self.is_fitted = True

        # 5. Save to Cache
        os.makedirs(self.working_dir, exist_ok=True)
        print(f"Saving OOF predictions to {oof_path}")
        np.save(oof_path, oof_preds)

        print(f"Saving Stage 1 Model to {model_path}")
        joblib.dump(self.model, model_path)

        return oof_preds

    def predict(self, X):
        """
        Predicts using the fitted model.

        Args:
            X (scipy.sparse.csr_matrix): Feature matrix.

        Returns:
            np.array: Predictions.
        """
        if not self.is_fitted:
            # Attempt to load if not in memory
            _, model_path = self._get_paths()
            if os.path.exists(model_path):
                print(f"Loading Stage 1 Model from {model_path} for inference...")
                self.model = joblib.load(model_path)
                self.is_fitted = True
            else:
                raise RuntimeError("Model is not fitted and no cached model found.")

        return self.model.predict(X)
