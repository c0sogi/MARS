import copy
import lightgbm as lgb
import xgboost as xgb
from library.config import Config


class BaseWrapper:
    """
    Base wrapper to ensure consistent interface for all ensemble models.
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        """
        Returns class probabilities.
        """
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Returns class predictions.
        """
        return self.model.predict(X)

    def get_feature_importances(self):
        """
        Returns feature importances if available.
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class LGBMWrapper(BaseWrapper):
    """
    Wrapper for LightGBM to handle callbacks and early stopping consistently.
    """

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        callbacks = []

        # Configure Early Stopping via callbacks
        if early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )

        # Suppress evaluation logging
        callbacks.append(lgb.log_evaluation(period=0))

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)


class XGBWrapper(BaseWrapper):
    """
    Wrapper for XGBoost to handle verbose flags and evaluation sets.
    """

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # XGBoost requires eval_set to be present for early_stopping_rounds
        if early_stopping_rounds is not None and eval_set is not None:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model.fit(X, y, verbose=False)


class ModelFactory:
    """
    Factory class to instantiate models for the Tri-Ensemble.
    """

    @staticmethod
    def get_model(model_name, epochs=None):
        """
        Creates and returns a model wrapper based on the name and configuration.

        Args:
            model_name (str): One of 'lgbm', 'xgb', 'cat'.
            epochs (int, optional): Overrides the default n_estimators/iterations.
                                    Useful for switching between Scout (mining) and Expert (final) modes.

        Returns:
            BaseWrapper: An instance of the model wrapper.
        """
        if model_name == "lgbm":
            params = copy.deepcopy(Config.LGBM_PARAMS)
            if epochs:
                params["n_estimators"] = epochs

            # Initialize LightGBM
            model = lgb.LGBMClassifier(**params)
            return LGBMWrapper(model)

        elif model_name == "xgb":
            params = copy.deepcopy(Config.XGB_PARAMS)
            if epochs:
                params["n_estimators"] = epochs

            # Initialize XGBoost
            model = xgb.XGBClassifier(**params)
            return XGBWrapper(model)

        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Supported: 'lgbm', 'xgb'."
            )
