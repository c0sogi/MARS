import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("model_zoo")


class LGBMExpert:
    """
    Wrapper for LightGBM model using the native API.
    """

    def __init__(self, params=None):
        self.params = params if params else Config.LGBM_PARAMS.copy()
        self.model = None
        self.num_boost_round = Config.TRAINING["EXPERT_EPOCHS"]
        self.early_stopping_rounds = Config.TRAINING["EARLY_STOPPING_ROUNDS"]
        self.verbose_eval = Config.TRAINING["VERBOSE_EVAL"]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with early stopping.
        """
        logger.info("Training LightGBM Expert...")

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=True
            ),
            lgb.log_evaluation(period=self.verbose_eval),
        ]

        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(
            f"LGBM training finished. Best iteration: {self.model.best_iteration}"
        )

    def predict_proba(self, X):
        """
        Predicts probabilities for class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        # LightGBM predict returns raw probabilities for binary classification
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class XGBExpert:
    """
    Wrapper for XGBoost model using the native API.
    """

    def __init__(self, params=None):
        self.params = params if params else Config.XGB_PARAMS.copy()
        self.model = None
        self.num_boost_round = Config.TRAINING["EXPERT_EPOCHS"]
        self.early_stopping_rounds = Config.TRAINING["EARLY_STOPPING_ROUNDS"]
        self.verbose_eval = Config.TRAINING["VERBOSE_EVAL"]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model with early stopping.
        """
        logger.info("Training XGBoost Expert...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

        logger.info(
            f"XGBoost training finished. Best iteration: {self.model.best_iteration}"
        )

    def predict_proba(self, X):
        """
        Predicts probabilities for class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )


class CatBoostExpert:
    """
    Wrapper for CatBoost model.
    """

    def __init__(self, params=None):
        self.params = params if params else Config.CATBOOST_PARAMS.copy()
        # Ensure iterations is set based on config epochs
        self.params["iterations"] = Config.TRAINING["EXPERT_EPOCHS"]
        self.params["early_stopping_rounds"] = Config.TRAINING["EARLY_STOPPING_ROUNDS"]
        # CatBoost uses 'verbose' int for logging period
        self.params["verbose"] = Config.TRAINING["VERBOSE_EVAL"]
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the CatBoost model.
        """
        logger.info("Training CatBoost Expert...")

        train_pool = Pool(X_train, label=y_train)
        eval_pool = None

        if X_val is not None and y_val is not None:
            eval_pool = Pool(X_val, label=y_val)

        self.model = CatBoostClassifier(**self.params)

        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        logger.info(
            f"CatBoost training finished. Best iteration: {self.model.get_best_iteration()}"
        )

    def predict_proba(self, X):
        """
        Predicts probabilities for class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        # CatBoost predict_proba returns (N, 2), we want column 1
        return self.model.predict_proba(X)[:, 1]


class EnsemblePredictor:
    """
    Manages the ensemble of heterogeneous models.
    """

    def __init__(self, models):
        """
        Args:
            models: List of trained model instances (LGBMExpert, XGBExpert, CatBoostExpert)
        """
        self.models = models

    def predict(self, X):
        """
        Computes the unweighted average of probabilities from all models.
        """
        if not self.models:
            raise ValueError("No models provided to EnsemblePredictor.")

        logger.info(f"Ensemble prediction using {len(self.models)} models...")

        preds_sum = np.zeros(X.shape[0])

        for i, model in enumerate(self.models):
            p = model.predict_proba(X)
            preds_sum += p

        return preds_sum / len(self.models)
