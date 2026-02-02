import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from library.config import Config


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor using Ridge Regression.
    Generates Out-Of-Fold (OOF) predictions to be used as features for Stage 2,
    and trains a final model for test set inference.
    """

    def __init__(self):
        self.model = Ridge(**Config.RIDGE_PARAMS)
        self.is_fitted = False

    def get_oof_predictions(self, X, y, load_cached_data: bool = True) -> np.ndarray:
        """
        Generates OOF predictions using K-Fold CV.
        Implements caching for the OOF array.
        Also fits the final model on the full dataset.

        Args:
            X: Sparse matrix of features (n_samples, n_features).
            y: Target array (n_samples,).
            load_cached_data: Whether to load OOF predictions from cache.

        Returns:
            np.ndarray: OOF predictions aligned with X.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        oof_cache_path = os.path.join(Config.WORKING_DIR, "stage1_oof_preds.npy")
        model_cache_path = os.path.join(Config.WORKING_DIR, "stage1_ridge_model.joblib")

        # 1. Try loading OOF from cache
        if load_cached_data and os.path.exists(oof_cache_path):
            print(f"Loading cached Stage 1 OOF predictions from {oof_cache_path}")
            try:
                oof_preds = np.load(oof_cache_path)
                if len(oof_preds) == X.shape[0]:
                    # If OOF is cached, we still need to ensure the model is fitted/loaded
                    if os.path.exists(model_cache_path):
                        self.load(
                            os.path.join(Config.WORKING_DIR, "stage1_ridge_model")
                        )
                    else:
                        print(
                            "Cached OOF found but model missing. Refitting final model..."
                        )
                        self.model.fit(X, y)
                        self.is_fitted = True
                        self.save(
                            os.path.join(Config.WORKING_DIR, "stage1_ridge_model")
                        )
                    return oof_preds
                else:
                    print(
                        f"OOF cache size mismatch ({len(oof_preds)} vs {X.shape[0]}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading OOF cache: {e}. Recomputing...")

        # 2. Compute OOF from scratch
        print(f"Generating Stage 1 OOF predictions with {Config.N_FOLDS}-Fold CV...")
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        oof_preds = np.zeros(X.shape[0])

        # Convert y to numpy array if it's a Series
        y_array = y.values if isinstance(y, pd.Series) else y

        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # Train fold model
            fold_model = Ridge(**Config.RIDGE_PARAMS)
            fold_model.fit(X_train, y_train)

            # Predict
            preds = fold_model.predict(X_val)
            oof_preds[val_idx] = preds

            # Metric
            fold_mae = mean_absolute_error(y_val, preds)
            fold_maes.append(fold_mae)
            print(f"Fold {fold+1} MAE: {fold_mae}")

        avg_mae = np.mean(fold_maes)
        print(f"Stage 1 CV MAE: {avg_mae}")

        # 3. Save OOF to cache
        print(f"Saving Stage 1 OOF predictions to {oof_cache_path}")
        np.save(oof_cache_path, oof_preds)

        # 4. Fit final model on all data
        print("Fitting final Stage 1 model on full dataset...")
        self.model.fit(X, y)
        self.is_fitted = True
        self.save(os.path.join(Config.WORKING_DIR, "stage1_ridge_model"))

        return oof_preds

    def predict(self, X) -> np.ndarray:
        """
        Predicts using the fitted final model.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Stage1Ridge model is not fitted. Call get_oof_predictions (which fits final model) or load first."
            )
        return self.model.predict(X)

    def save(self, base_path_without_ext: str):
        """
        Saves the Ridge model.
        """
        path = f"{base_path_without_ext}.joblib"
        joblib.dump(self.model, path)
        print(f"Stage 1 model saved to {path}")

    def load(self, base_path_without_ext: str):
        """
        Loads the Ridge model.
        """
        path = f"{base_path_without_ext}.joblib"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage 1 model not found at {path}")
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Stage 1 model loaded from {path}")


class Stage2LGBM:
    """
    Stage 2: Multi-Resolution Gradient Booster using LightGBM.
    Refines predictions using stacked features and minimizes MAE.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
    ):
        """
        Trains the LightGBM model with early stopping.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            feature_cols: List of feature column names.
            target_col: Name of the target column.
        """
        print("Preparing Stage 2 datasets...")

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        print("Starting LightGBM training...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.LGBM_VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Print final validation metric
        # LightGBM records evaluation results in the booster
        # We access the best score for the validation set
        if self.model.best_score:
            val_score = self.model.best_score.get("valid", {}).get(
                "l1"
            )  # 'l1' corresponds to 'mae' metric
            if val_score is not None:
                print(f"Final Validation MAE: {val_score}")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Generates predictions using the trained model.
        """
        if self.model is None:
            raise RuntimeError("Stage2LGBM model is not fitted.")

        return self.model.predict(
            df[feature_cols], num_iteration=self.model.best_iteration
        )

    def save(self, base_path_without_ext: str):
        """
        Saves the LightGBM model.
        """
        path = f"{base_path_without_ext}.txt"
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model.")
        self.model.save_model(path)
        print(f"Stage 2 model saved to {path}")

    def load(self, base_path_without_ext: str):
        """
        Loads the LightGBM model.
        """
        path = f"{base_path_without_ext}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage 2 model not found at {path}")
        self.model = lgb.Booster(model_file=path)
        print(f"Stage 2 model loaded from {path}")
