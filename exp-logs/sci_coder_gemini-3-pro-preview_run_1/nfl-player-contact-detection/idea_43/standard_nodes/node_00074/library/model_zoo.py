import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import KADM_CONFIG
from library.utils import setup_logger, seed_everything

# Setup logger
logger = setup_logger(name="model_zoo")


def lgb_mcc_score(preds, data):
    """
    Custom MCC metric for LightGBM.
    """
    y_true = data.get_label()
    # Standard threshold of 0.5 for metric calculation during training
    y_pred = (preds > 0.5).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    return "mcc", score, True


def xgb_mcc_score(preds, dtrain):
    """
    Custom MCC metric for XGBoost.
    """
    y_true = dtrain.get_label()
    y_pred = (preds > 0.5).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    return "mcc", score


class LGBMWrapper:
    """
    Wrapper for LightGBM model with custom MCC metric and specific configuration.
    """

    def __init__(self, config=KADM_CONFIG):
        self.params = config["models"]["lgbm"].copy()
        self.train_params = config["training"]
        self.model = None
        self.seed = self.params.get("seed", 42)

        # Remove params not needed for constructor if any,
        # but lgb.train handles dicts well.
        # 'n_jobs' is in params.

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.
        """
        seed_everything(self.seed)

        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")

        # Handle metric configuration
        # If metric is "None" string in config, we remove it to rely on feval or default
        if self.params.get("metric") == "None":
            del self.params["metric"]

        logger.info("Training LightGBM Expert...")

        callbacks = [
            lgb.log_evaluation(period=self.train_params["verbose_eval"]),
            lgb.early_stopping(
                stopping_rounds=self.train_params["early_stopping_rounds"]
            ),
        ]

        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.train_params["num_boost_round"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=lgb_mcc_score,
            callbacks=callbacks,
        )

        # Log final validation score
        if X_val is not None:
            preds = self.model.predict(X_val)
            score = matthews_corrcoef(y_val, (preds > 0.5).astype(int))
            logger.info(f"LGBM Final Validation MCC: {score}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("LGBM model not trained yet.")
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the model using joblib.
        """
        joblib.dump(self.model, path)
        logger.info(f"LGBM model saved to {path}")

    def load(self, path):
        """
        Loads the model using joblib.
        """
        self.model = joblib.load(path)
        logger.info(f"LGBM model loaded from {path}")


class XGBWrapper:
    """
    Wrapper for XGBoost model with custom MCC metric and specific configuration.
    """

    def __init__(self, config=KADM_CONFIG):
        self.params = config["models"]["xgb"].copy()
        self.train_params = config["training"]
        self.model = None
        self.seed = self.params.get("random_state", 42)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.
        """
        seed_everything(self.seed)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        logger.info("Training XGBoost Expert...")

        # Sanitize verbose_eval for XGBoost (requires False for silence, not -1)
        verbose_eval = self.train_params["verbose_eval"]
        if isinstance(verbose_eval, int) and verbose_eval <= 0:
            verbose_eval = False

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.train_params["num_boost_round"],
            evals=evals,
            custom_metric=xgb_mcc_score,
            early_stopping_rounds=self.train_params["early_stopping_rounds"],
            verbose_eval=verbose_eval,
        )

        # Log final validation score
        if X_val is not None:
            preds = self.model.predict(dval)
            score = matthews_corrcoef(y_val, (preds > 0.5).astype(int))
            logger.info(f"XGB Final Validation MCC: {score}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("XGB model not trained yet.")

        # XGBoost requires DMatrix for prediction if trained with train()
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path):
        """
        Saves the model using joblib.
        """
        joblib.dump(self.model, path)
        logger.info(f"XGB model saved to {path}")

    def load(self, path):
        """
        Loads the model using joblib.
        """
        self.model = joblib.load(path)
        logger.info(f"XGB model loaded from {path}")


class DualModelEnsemble:
    """
    Orchestrates the Dual-Ensemble (LGBM + XGB).
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.lgbm = LGBMWrapper(config)
        self.xgb = XGBWrapper(config)
        self.model_dir = config["paths"]["model_dir"]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains both models sequentially.
        """
        logger.info("Starting Dual-Ensemble Training...")

        # Train LGBM
        self.lgbm.fit(X_train, y_train, X_val, y_val)

        # Train XGB
        self.xgb.fit(X_train, y_train, X_val, y_val)

        logger.info("Dual-Ensemble Training Completed.")

    def predict(self, X):
        """
        Generates ensemble predictions (unweighted average).
        """
        logger.info("Generating ensemble predictions...")

        pred_lgbm = self.lgbm.predict(X)
        pred_xgb = self.xgb.predict(X)

        # Unweighted average
        ensemble_pred = (pred_lgbm + pred_xgb) / 2.0
        return ensemble_pred

    def save(self, directory=None):
        """
        Saves both models to the specified directory (or config default).
        """
        if directory is None:
            directory = self.model_dir

        os.makedirs(directory, exist_ok=True)

        lgbm_path = os.path.join(directory, "expert_lgbm.joblib")
        xgb_path = os.path.join(directory, "expert_xgb.joblib")

        self.lgbm.save(lgbm_path)
        self.xgb.save(xgb_path)

    def load(self, directory=None):
        """
        Loads both models from the specified directory (or config default).
        """
        if directory is None:
            directory = self.model_dir

        lgbm_path = os.path.join(directory, "expert_lgbm.joblib")
        xgb_path = os.path.join(directory, "expert_xgb.joblib")

        if os.path.exists(lgbm_path):
            self.lgbm.load(lgbm_path)
        else:
            logger.warning(f"LGBM model not found at {lgbm_path}")

        if os.path.exists(xgb_path):
            self.xgb.load(xgb_path)
        else:
            logger.warning(f"XGB model not found at {xgb_path}")
