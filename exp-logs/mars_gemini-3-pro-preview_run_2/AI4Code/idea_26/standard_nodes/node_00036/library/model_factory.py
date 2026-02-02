import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from library.config import Config
from library.utils import save_joblib, load_joblib, set_seed


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor (Ridge Regression).
    Acts as the 'Signpost' model, mapping explicit keywords to rank positions.
    """

    def __init__(self):
        self.config = Config
        self.model_path = self.config.CACHE_RIDGE_MODEL
        self.oof_path = self.config.CACHE_STAGE1_OOF
        self.model = None
        set_seed(self.config.SEED)

    def fit(self, X, y):
        """
        Fits the Ridge model on the full dataset and saves it.
        """
        print(f"Training Stage 1 Ridge Model with params: {self.config.RIDGE_PARAMS}")
        self.model = Ridge(**self.config.RIDGE_PARAMS)
        self.model.fit(X, y)

        print(f"Saving Stage 1 model to {self.model_path}")
        save_joblib(self.model, self.model_path)

    def predict(self, X):
        """
        Predicts using the trained Ridge model. Loads from disk if necessary.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 1 model from {self.model_path}")
                self.model = load_joblib(self.model_path)
            else:
                raise FileNotFoundError("Stage 1 model not found. Call fit() first.")

        return self.model.predict(X)

    def get_oof_predictions(self, X, y, groups=None, load_cached_data=True):
        """
        Generates Out-Of-Fold (OOF) predictions for the training set.
        Uses GroupKFold if groups are provided to prevent leakage.

        Args:
            X: Sparse feature matrix.
            y: Target array.
            groups: Group labels (ancestor_id) for GroupKFold.
            load_cached_data: Whether to try loading from cache.

        Returns:
            np.array: OOF predictions aligned with X.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.oof_path):
            print(f"Loading cached Stage 1 OOF predictions from {self.oof_path}")
            df_oof = pd.read_parquet(self.oof_path)
            return df_oof["pred"].values

        # 2. Compute OOF
        print("Generating Stage 1 OOF predictions...")

        if groups is not None:
            kf = GroupKFold(n_splits=self.config.N_FOLDS)
            split_iter = kf.split(X, y, groups)
        else:
            kf = KFold(
                n_splits=self.config.N_FOLDS,
                shuffle=True,
                random_state=self.config.SEED,
            )
            split_iter = kf.split(X, y)

        oof_preds = np.zeros(len(y))

        # We assume X is scipy.sparse matrix or numpy array that supports slicing
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            model = Ridge(**self.config.RIDGE_PARAMS)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)

        # 3. Save to cache
        print(f"Saving Stage 1 OOF predictions to {self.oof_path}")
        os.makedirs(os.path.dirname(self.oof_path), exist_ok=True)
        pd.DataFrame({"pred": oof_preds}).to_parquet(self.oof_path, index=False)

        return oof_preds


class Stage2LGBM:
    """
    Stage 2: Decoupled Dual-View Gradient Booster (LightGBM).
    Refines predictions using neighborhood aggregation features and interaction terms.
    """

    def __init__(self):
        self.config = Config
        self.model_path = self.config.CACHE_LGBM_MODEL
        self.model = None
        set_seed(self.config.SEED)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fits the LightGBM model with Early Stopping.
        """
        print(f"Training Stage 2 LightGBM Model with params: {self.config.LGBM_PARAMS}")

        self.model = LGBMRegressor(**self.config.LGBM_PARAMS)

        callbacks = [
            early_stopping(stopping_rounds=self.config.EARLY_STOPPING_ROUNDS),
            log_evaluation(period=self.config.VERBOSE_EVAL),
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=callbacks,
        )

        print(f"Saving Stage 2 model to {self.model_path}")
        # LightGBM saves best iteration automatically when using early stopping
        # However, to persist to disk we use the booster's save_model
        # Note: self.model is the sklearn wrapper.
        # We can save the booster or pickle the wrapper.
        # The Config points to a .txt file, implying booster.save_model format.

        booster = self.model.booster_
        booster.save_model(self.model_path)

    def predict(self, X):
        """
        Predicts using the trained LightGBM model.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 2 model from {self.model_path}")
                # Load into a fresh LGBMRegressor container
                self.model = LGBMRegressor(**self.config.LGBM_PARAMS)
                # Load the booster
                booster = (
                    joblib.load(self.model_path)
                    if not self.model_path.endswith(".txt")
                    else None
                )

                if booster is None:
                    # If it's a text file, load via booster init
                    import lightgbm as lgb

                    self.model._Booster = lgb.Booster(model_file=self.model_path)
                    self.model.fitted_ = True

                    # Manually restore feature count metadata for sklearn validation
                    n_features = self.model._Booster.num_feature()
                    self.model._n_features = n_features
                    self.model.n_features_in_ = n_features
                else:
                    # Fallback if we decided to use joblib (though config implies txt)
                    self.model = booster
            else:
                raise FileNotFoundError("Stage 2 model not found. Call fit() first.")

        return self.model.predict(X)
