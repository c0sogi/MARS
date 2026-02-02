import os
import logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from library.config import MODEL_PARAMS, N_JOBS, SEED
from library.utils import save_artifact, load_artifact, setup_logging

# Initialize logging
setup_logging()


class ModelFactory:
    """
    Factory class to instantiate specific Gradient Boosting models
    configured for Soft Target Regression (Binary Objective).
    """

    @staticmethod
    def create_model(model_type):
        """
        Creates a model instance based on the type string.

        Args:
            model_type (str): 'lgbm', 'xgb', or 'cat'.

        Returns:
            The instantiated regressor model.
        """
        if model_type not in MODEL_PARAMS:
            raise ValueError(f"Unknown model type: {model_type}")

        params = MODEL_PARAMS[model_type].copy()

        if model_type == "lgbm":
            # LGBMRegressor with binary objective handles soft targets [0,1]
            return lgb.LGBMRegressor(**params)

        elif model_type == "xgb":
            # XGBRegressor with binary:logistic handles soft targets [0,1]
            return xgb.XGBRegressor(**params)

        elif model_type == "cat":
            # CatBoostRegressor with Logloss handles soft targets [0,1]
            return CatBoostRegressor(**params)

        else:
            raise ValueError(f"Model type {model_type} not implemented.")


class TriEnsemble:
    """
    Unified Heterogeneous Tri-Ensemble.
    Manages LightGBM, XGBoost, and CatBoost models as a single entity.
    """

    def __init__(self):
        self.models = {
            "lgbm": ModelFactory.create_model("lgbm"),
            "xgb": ModelFactory.create_model("xgb"),
            "cat": ModelFactory.create_model("cat"),
        }
        self.model_names = ["lgbm", "xgb", "cat"]

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Trains all three models in the ensemble.

        Args:
            X_train: Training features.
            y_train: Training soft targets.
            X_val: Validation features (optional).
            y_val: Validation soft targets (optional).
            early_stopping_rounds (int): Rounds for early stopping.
        """
        logging.info("Starting Tri-Ensemble Training...")

        for name, model in self.models.items():
            logging.info(f"Training {name.upper()}...")

            try:
                if name == "lgbm":
                    callbacks = []
                    if early_stopping_rounds is not None and X_val is not None:
                        callbacks.append(
                            lgb.early_stopping(
                                stopping_rounds=early_stopping_rounds, verbose=False
                            )
                        )
                        callbacks.append(lgb.log_evaluation(period=0))  # Silence

                    eval_set = [(X_val, y_val)] if X_val is not None else None

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        eval_metric="binary_logloss",
                        callbacks=callbacks,
                    )

                elif name == "xgb":
                    eval_set = [(X_val, y_val)] if X_val is not None else None

                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

                elif name == "cat":
                    eval_set = (X_val, y_val) if X_val is not None else None

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=(
                            early_stopping_rounds if X_val is not None else None
                        ),
                        verbose=False,
                    )

                # Log validation score if available
                if X_val is not None:
                    # Predict to check score (sanity check)
                    preds = model.predict(X_val)
                    # Simple LogLoss check
                    eps = 1e-15
                    preds = np.clip(preds, eps, 1 - eps)
                    # Assuming y_val might be soft, but logloss calc usually expects binary or soft matches
                    # Just printing mean pred to ensure it's working
                    logging.info(
                        f"{name.upper()} trained. Mean prediction: {np.mean(preds):.4f}"
                    )

            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                raise e

        logging.info("Tri-Ensemble Training Complete.")

    def predict(self, X):
        """
        Generates averaged predictions from the ensemble.

        Args:
            X: Features.

        Returns:
            np.ndarray: Averaged probability of contact [0, 1].
        """
        preds_accum = np.zeros(len(X))

        for name, model in self.models.items():
            # Regressors with binary/logistic objective return probabilities directly
            p = model.predict(X)
            preds_accum += p

        # Unweighted Average
        return preds_accum / len(self.models)

    def save(self, directory):
        """
        Saves the individual models to the specified directory.

        Args:
            directory (str): Path to save models.
        """
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}_model.joblib")
            save_artifact(model, path)
        logging.info(f"Tri-Ensemble saved to {directory}")

    def load(self, directory):
        """
        Loads the individual models from the specified directory.

        Args:
            directory (str): Path to load models from.
        """
        for name in self.model_names:
            path = os.path.join(directory, f"{name}_model.joblib")
            if os.path.exists(path):
                model = load_artifact(path)
                if model is not None:
                    self.models[name] = model
            else:
                logging.warning(
                    f"Model file {path} not found. Keeping initialized model."
                )
        logging.info(f"Tri-Ensemble loaded from {directory}")
