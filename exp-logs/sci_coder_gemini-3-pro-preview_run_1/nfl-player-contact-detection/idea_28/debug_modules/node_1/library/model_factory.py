import os
import numpy as np
import joblib
import logging
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from library.config import Config

# Setup logger
logger = logging.getLogger("NFL_Contact_Detection")


class TriModelEnsemble:
    """
    A Unified Heterogeneous Tri-Ensemble wrapper containing:
    1. LightGBM (Leaf-wise)
    2. XGBoost (Level-wise)
    3. HistGradientBoosting (Symmetric/Oblivious tree approximation)
    """

    def __init__(self):
        """
        Initialize the three model instances with configurations from Config.
        """
        self.models = {}
        self.model_names = ["lgbm", "xgb", "hgb"]

        # Initialize LightGBM
        # Note: We filter out 'is_unbalance' if it causes issues in init,
        # but LGBMClassifier supports it.
        self.models["lgbm"] = lgb.LGBMClassifier(**Config.LGBM_PARAMS)

        # Initialize XGBoost
        self.models["xgb"] = xgb.XGBClassifier(**Config.XGB_PARAMS)

        # Initialize HistGradientBoosting (CatBoost substitute)
        self.models["hgb"] = HistGradientBoostingClassifier(**Config.SKLEARN_HGB_PARAMS)

        self.is_fitted = {name: False for name in self.model_names}

    def fit(self, X_train, y_train, X_val=None, y_val=None, specific_model=None):
        """
        Trains the ensemble or a specific model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional, used for early stopping).
            y_val: Validation labels (optional).
            specific_model (str): If provided ('lgbm', 'xgb', 'hgb'), only trains that model.
                                  Otherwise, trains all.
        """
        target_models = [specific_model] if specific_model else self.model_names

        for name in target_models:
            if name not in self.models:
                logger.warning(f"Model {name} not found in ensemble. Skipping.")
                continue

            logger.info(f"Training {name.upper()}...")
            model = self.models[name]

            try:
                if name == "lgbm":
                    # LightGBM with Early Stopping
                    callbacks = []
                    if X_val is not None and y_val is not None:
                        callbacks.append(
                            lgb.early_stopping(
                                stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                                verbose=False,
                            )
                        )
                        eval_set = [(X_val, y_val)]
                    else:
                        eval_set = None

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        eval_metric="average_precision",
                        callbacks=callbacks,
                    )

                    # Log validation score if available
                    if eval_set:
                        # Best score is stored in best_score_ attribute usually
                        # But for safety we can predict and print
                        val_pred = model.predict_proba(X_val)[:, 1]
                        logger.info(
                            f"{name.upper()} Validation Mean Pred: {np.mean(val_pred)}"
                        )

                elif name == "xgb":
                    # XGBoost with Early Stopping
                    eval_set = None
                    if X_val is not None and y_val is not None:
                        eval_set = [(X_val, y_val)]

                    # XGBoost fit parameters for early stopping
                    fit_params = {"verbose": False}
                    if eval_set:
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = (
                            Config.EARLY_STOPPING_ROUNDS
                        )

                    model.fit(X_train, y_train, **fit_params)

                    if eval_set:
                        val_pred = model.predict_proba(X_val)[:, 1]
                        logger.info(
                            f"{name.upper()} Validation Mean Pred: {np.mean(val_pred)}"
                        )

                elif name == "hgb":
                    # HistGradientBoostingClassifier
                    # Does not support external eval_set for early stopping in fit()
                    # It uses internal validation_fraction if configured, or early_stopping param.
                    model.fit(X_train, y_train)

                    if X_val is not None:
                        val_score = model.score(X_val, y_val)
                        logger.info(f"{name.upper()} Validation Accuracy: {val_score}")

                self.is_fitted[name] = True
                logger.info(f"Finished training {name.upper()}.")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                raise e

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input samples.
        Returns the unweighted average of all trained models.

        Args:
            X: Input features.

        Returns:
            np.array: Array of probabilities for class 1.
        """
        preds = []
        trained_count = 0

        for name, model in self.models.items():
            if self.is_fitted[name]:
                # predict_proba returns (N, 2), we take column 1
                p = model.predict_proba(X)[:, 1]
                preds.append(p)
                trained_count += 1

        if trained_count == 0:
            raise RuntimeError("No models have been fitted yet.")

        # Average predictions
        avg_preds = np.mean(preds, axis=0)
        return avg_preds

    def predict_individual(self, X):
        """
        Returns predictions from individual models.
        Used for Hard Negative Mining (Union of high-confidence errors).

        Args:
            X: Input features.

        Returns:
            dict: {model_name: probabilities_array}
        """
        results = {}
        for name, model in self.models.items():
            if self.is_fitted[name]:
                results[name] = model.predict_proba(X)[:, 1]
            else:
                logger.warning(f"Attempting to predict with unfitted model {name}")
                results[name] = np.zeros(X.shape[0])
        return results

    def save(self, directory):
        """
        Saves the trained models to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            if self.is_fitted[name]:
                path = os.path.join(directory, f"{name}_model.joblib")
                joblib.dump(model, path)
                logger.info(f"Saved {name} to {path}")

    def load(self, directory):
        """
        Loads trained models from the specified directory.
        """
        for name in self.models.keys():
            path = os.path.join(directory, f"{name}_model.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                self.is_fitted[name] = True
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"Model file {path} not found.")
