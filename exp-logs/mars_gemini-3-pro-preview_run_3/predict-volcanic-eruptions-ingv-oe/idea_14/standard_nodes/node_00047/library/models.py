import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything, compute_mae

# Ensure reproducibility
seed_everything(Config.SEED)


class LGBMEnsemble:
    """
    A simplified robust ensemble using only LightGBM with Stratified K-Fold CV.
    Replaces the complex stacking architecture that failed due to weak base learners (Cite solution_lesson_node_00046).
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.models = []
        self.model_path = os.path.join(Config.WORKING_DIR, "lgbm_ensemble.pkl")

    def _get_stratified_folds(self, y, n_splits):
        # Create bins for stratification
        num_bins = min(10, len(np.unique(y)))
        if num_bins < 2:
            from sklearn.model_selection import KFold

            return KFold(
                n_splits=n_splits, shuffle=True, random_state=Config.SEED
            ).split(np.zeros(len(y)), y)

        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        return skf.split(np.zeros(len(y)), y_bins)

    def fit(self, X, y):
        """
        Trains 5 LightGBM models on stratified folds.
        """
        print(f"Starting LightGBM Ensemble Training ({Config.N_FOLDS} folds)...")
        self.models = []

        folds = list(self._get_stratified_folds(y, Config.N_FOLDS))
        oof_preds = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"  Processing Fold {fold_idx + 1}/{Config.N_FOLDS}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**self.params)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=0),
            ]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=callbacks,
            )

            self.models.append(model)
            oof_preds[val_idx] = model.predict(X_val)
            print(f"    Fold {fold_idx+1} Best Iteration: {model.best_iteration_}")

        mae = compute_mae(y, oof_preds)
        print(f"\nEnsemble OOF MAE: {mae}")
        self.save_models()

    def predict(self, X):
        """
        Averages predictions from all fold models.
        """
        if not self.models:
            raise RuntimeError("Models not trained.")

        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X)

        return preds / len(self.models)

    def save_models(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.models, self.model_path)
        print(f"Models saved to {self.model_path}")

    def load_models(self):
        if os.path.exists(self.model_path):
            self.models = joblib.load(self.model_path)
            print(f"Models loaded from {self.model_path}")
            return True
        return False
