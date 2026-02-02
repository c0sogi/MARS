import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import os
import copy
from library.config import Config
from library.utils import save_model, load_model, Timer


class BaseModel:
    """
    Abstract base class for models in the ensemble.
    """

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class LGBMModel(BaseModel):
    """
    Wrapper for LightGBM model (Leaf-wise growth).
    """

    def __init__(self, params=None):
        self.params = params if params else Config.LGBM_PARAMS.copy()
        self.model = None
        self.best_iteration = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        with Timer("LGBM Training"):
            train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            valid_sets = [train_ds]
            valid_names = ["train"]

            if X_val is not None and y_val is not None:
                val_ds = lgb.Dataset(
                    X_val, label=y_val, feature_name=feature_names, reference=train_ds
                )
                valid_sets.append(val_ds)
                valid_names.append("valid")

            # Setup callbacks for early stopping and logging
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ]

            self.model = lgb.train(
                self.params,
                train_ds,
                num_boost_round=self.params.get("n_estimators", 2000),
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

            self.best_iteration = self.model.best_iteration

            if X_val is not None:
                # Retrieve best score with full precision
                # Metric is 'auc' per config
                score = self.model.best_score["valid"]["auc"]
                print(f"LGBM Best Validation AUC: {score}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("LGBM model not trained.")
        return self.model.predict(X, num_iteration=self.best_iteration)


class XGBModel(BaseModel):
    """
    Wrapper for XGBoost model (Level-wise growth).
    """

    def __init__(self, params=None):
        self.params = params if params else Config.XGB_PARAMS.copy()
        self.model = None
        self.best_iteration = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        with Timer("XGB Training"):
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            evals = [(dtrain, "train")]

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
                evals.append((dval, "valid"))

            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get("n_estimators", 2000),
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=100,
            )

            self.best_iteration = self.model.best_iteration

            if X_val is not None:
                # Print best score with full precision
                print(f"XGB Best Validation Score: {self.model.best_score}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("XGB model not trained.")
        # XGBoost requires DMatrix
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))


class LGBMDartModel(LGBMModel):
    """
    LightGBM with DART boosting to substitute for CatBoost.
    Provides structural diversity via Dropout Multiple Additive Regression Trees.
    """

    def __init__(self):
        params = Config.LGBM_PARAMS.copy()
        # DART specific configuration
        params.update(
            {
                "boosting_type": "dart",
                "drop_rate": 0.1,
                "skip_drop": 0.5,
                "xgboost_dart_mode": True,
                "metric": "auc",
            }
        )
        super().__init__(params)


class TriEnsemble:
    """
    Heterogeneous Ensemble containing LGBM, XGB, and LGBM-DART models.
    Manages feature selection, training, and ensemble averaging.
    """

    def __init__(self):
        self.lgbm = LGBMModel()
        self.xgb = XGBModel()
        self.dart = LGBMDartModel()
        self.models = {"lgbm": self.lgbm, "xgb": self.xgb, "dart": self.dart}
        self.feature_cols = None

    def _get_feature_cols(self, df):
        """
        Identifies feature columns by excluding known metadata.
        """
        exclude_cols = {
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "soft_contact",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "datetime",
            "group_id",
            "p2_int",
            "game_key",
            "play_id",
        }
        # Select numeric columns not in exclude list
        candidates = [c for c in df.columns if c not in exclude_cols]
        # Verify numeric types
        numeric_candidates = (
            df[candidates].select_dtypes(include=[np.number]).columns.tolist()
        )
        return numeric_candidates

    def fit(self, train_df, val_df):
        """
        Fits all three models in the ensemble using the provided dataframes.
        """
        self.feature_cols = self._get_feature_cols(train_df)
        print(f"Selected {len(self.feature_cols)} features for training.")

        X_train = train_df[self.feature_cols]
        y_train = train_df["contact"]

        X_val = val_df[self.feature_cols]
        y_val = val_df["contact"]

        for name, model in self.models.items():
            print(f"\n--- Training {name.upper()} Model ---")
            model.fit(X_train, y_train, X_val, y_val, feature_names=self.feature_cols)

    def predict(self, df):
        """
        Generates averaged probability predictions from the ensemble.
        """
        if self.feature_cols is None:
            self.feature_cols = self._get_feature_cols(df)

        X = df[self.feature_cols]

        preds = []
        for name, model in self.models.items():
            p = model.predict(X)
            preds.append(p)

        # Unweighted average of probabilities
        avg_preds = np.mean(preds, axis=0)
        return avg_preds

    def save(self, directory):
        """
        Saves all models and the feature column list to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        # Save feature columns
        save_model(
            self.feature_cols, os.path.join(directory, "tri_ensemble_features.joblib")
        )

        # Save models
        for name, model in self.models.items():
            save_model(model.model, os.path.join(directory, f"{name}_model.joblib"))

    def load(self, directory):
        """
        Loads all models and the feature column list from the specified directory.
        """
        feat_path = os.path.join(directory, "tri_ensemble_features.joblib")
        if os.path.exists(feat_path):
            self.feature_cols = load_model(feat_path)

        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}_model.joblib")
            if os.path.exists(path):
                model.model = load_model(path)
                # Attempt to restore best_iteration if available in the loaded object
                if hasattr(model.model, "best_iteration"):
                    model.best_iteration = model.model.best_iteration
