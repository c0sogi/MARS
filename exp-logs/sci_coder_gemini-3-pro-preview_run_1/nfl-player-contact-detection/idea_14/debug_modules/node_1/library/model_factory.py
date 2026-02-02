import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Tuple

from library.config import ModelConfig, MODEL_OUTPUT_DIR, SEED
from library.utils import set_seed, compute_mcc

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


class BaseModel(ABC):
    """
    Abstract base class for all models in the ensemble.
    Enforces a consistent interface for training, prediction, and persistence.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        set_seed(SEED)

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Trains the model with early stopping.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts probabilities for the positive class.
        """
        pass

    def save(self, filename: str) -> None:
        """
        Saves the model to the MODEL_OUTPUT_DIR.
        """
        path = os.path.join(MODEL_OUTPUT_DIR, filename)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, filename: str) -> None:
        """
        Loads the model from the MODEL_OUTPUT_DIR.
        """
        path = os.path.join(MODEL_OUTPUT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")


class LGBMWrapper(BaseModel):
    """
    Wrapper for LightGBM model.
    Uses leaf-wise growth strategy and handles class imbalance via 'is_unbalance'.
    """

    def __init__(
        self, config: Optional[ModelConfig] = None, overrides: Optional[Dict] = None
    ):
        self.config = config if config is not None else ModelConfig()
        params = self.config.lgbm_params.copy()
        if overrides:
            params.update(overrides)
        super().__init__(params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:

        print(f"Training LightGBM with params: {self.params}")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks for logging and early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
            lgb.log_evaluation(period=self.config.verbose_eval),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log final validation metric
        val_preds = self.model.predict(X_val)
        # Convert probabilities to binary for MCC calculation (using 0.5 threshold temporarily for logging)
        val_preds_binary = (val_preds > 0.5).astype(int)
        mcc = compute_mcc(y_val, val_preds_binary)
        print(
            f"LightGBM Training Completed. Final Validation MCC (thresh=0.5): {mcc:.16f}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X)


class XGBWrapper(BaseModel):
    """
    Wrapper for XGBoost model.
    Uses level-wise growth (by default or configured) and 'scale_pos_weight'.
    """

    def __init__(
        self, config: Optional[ModelConfig] = None, overrides: Optional[Dict] = None
    ):
        self.config = config if config is not None else ModelConfig()
        params = self.config.xgb_params.copy()
        if overrides:
            params.update(overrides)
        super().__init__(params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:

        print(f"Training XGBoost with params: {self.params}")

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, "train"), (dval, "valid")]

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get("n_estimators", 1000),
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=self.config.verbose_eval,
        )

        # Log final validation metric
        val_preds = self.model.predict(dval)
        val_preds_binary = (val_preds > 0.5).astype(int)
        mcc = compute_mcc(y_val, val_preds_binary)
        print(
            f"XGBoost Training Completed. Final Validation MCC (thresh=0.5): {mcc:.16f}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # XGBoost requires DMatrix for predict if trained with train()
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


class EnsemblePredictor:
    """
    Manages the heterogeneous ensemble of LightGBM and XGBoost models.
    Handles loading models and generating averaged predictions.
    """

    def __init__(self, lgbm_path: Optional[str] = None, xgb_path: Optional[str] = None):
        self.lgbm_model = LGBMWrapper()
        self.xgb_model = XGBWrapper()

        if lgbm_path:
            self.lgbm_model.load(lgbm_path)
            self.has_lgbm = True
        else:
            self.has_lgbm = False

        if xgb_path:
            self.xgb_model.load(xgb_path)
            self.has_xgb = True
        else:
            self.has_xgb = False

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions from the ensemble.
        Returns the unweighted average of probabilities from available models.
        """
        preds_list = []

        if self.has_lgbm:
            lgbm_preds = self.lgbm_model.predict(X)
            preds_list.append(lgbm_preds)

        if self.has_xgb:
            xgb_preds = self.xgb_model.predict(X)
            preds_list.append(xgb_preds)

        if not preds_list:
            raise ValueError("No models loaded in EnsemblePredictor.")

        # Average predictions
        avg_preds = np.mean(preds_list, axis=0)
        return avg_preds
