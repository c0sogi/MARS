import logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from library import config, utils

# Attempt to import CatBoost; handle absence gracefully by flagging for fallback
try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class ModelFactory:
    """
    Factory class to create, train, and manage heterogeneous models
    (LightGBM, XGBoost, and CatBoost/HistGradientBoosting).
    Encapsulates library-specific API differences and hyperparameter management.
    """

    @staticmethod
    def train_model(model_type, X_train, y_train, X_val=None, y_val=None):
        """
        Trains a model of the specified type using parameters from config.py.

        Args:
            model_type (str): Identifier for the model ('lgbm', 'xgb', 'cat').
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame, optional): Validation features for early stopping.
            y_val (pd.Series, optional): Validation targets for early stopping.

        Returns:
            The trained model object.
        """
        logging.info(f"Initializing training for model type: {model_type.upper()}")

        # =========================================================================
        # 1. LightGBM
        # =========================================================================
        if model_type == "lgbm":
            params = config.LGBM_PARAMS.copy()

            # Initialize Classifier
            clf = lgb.LGBMClassifier(**params)

            # Setup callbacks for early stopping
            callbacks = []
            if config.EARLY_STOPPING_ROUNDS > 0 and X_val is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
                    )
                )
                callbacks.append(
                    lgb.log_evaluation(period=0)
                )  # Suppress verbose logging

            eval_set = [(X_val, y_val)] if X_val is not None else None

            clf.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric=params.get("metric", "binary_logloss"),
                callbacks=callbacks,
            )
            return clf

        # =========================================================================
        # 2. XGBoost
        # =========================================================================
        elif model_type == "xgb":
            params = config.XGB_PARAMS.copy()

            # Dynamic calculation of scale_pos_weight for imbalance handling
            if (
                "scale_pos_weight" not in params
                or params.get("scale_pos_weight") is None
            ):
                n_pos = np.sum(y_train == 1)
                n_neg = np.sum(y_train == 0)
                # Calculate ratio: negatives / positives
                scale_weight = n_neg / max(n_pos, 1)
                params["scale_pos_weight"] = scale_weight
                logging.info(
                    f"XGBoost: Dynamically calculated scale_pos_weight = {scale_weight:.4f}"
                )

            # Configure early stopping in constructor for XGBoost >= 1.6
            if X_val is not None and config.EARLY_STOPPING_ROUNDS > 0:
                params["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS

            clf = xgb.XGBClassifier(**params)

            eval_set = [(X_val, y_val)] if X_val is not None else None

            # Fit with early stopping
            clf.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            return clf

        # =========================================================================
        # 3. CatBoost (with Fallback to HistGradientBoosting)
        # =========================================================================
        elif model_type == "cat":
            if HAS_CATBOOST:
                params = config.CAT_PARAMS.copy()
                clf = cb.CatBoostClassifier(**params)

                eval_set = (X_val, y_val) if X_val is not None else None

                clf.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=(
                        config.EARLY_STOPPING_ROUNDS if X_val is not None else None
                    ),
                    verbose=False,
                )
                return clf
            else:
                logging.warning(
                    "CatBoost package not found. Falling back to sklearn HistGradientBoostingClassifier."
                )

                # Map CAT_PARAMS to HistGradientBoostingClassifier parameters
                cat_params = config.CAT_PARAMS
                hgb_params = {
                    "max_iter": cat_params.get("iterations", 2000),
                    "learning_rate": cat_params.get("learning_rate", 0.02),
                    "max_depth": cat_params.get("depth", 10),
                    "random_state": cat_params.get("random_seed", config.SEED),
                    "early_stopping": True,
                    "n_iter_no_change": config.EARLY_STOPPING_ROUNDS,
                    "verbose": 0,
                }

                # Map class weights
                if cat_params.get("auto_class_weights") == "Balanced":
                    hgb_params["class_weight"] = "balanced"

                clf = HistGradientBoostingClassifier(**hgb_params)

                # HGBC uses internal validation split or training loss for early stopping
                # It does not accept external eval_set in fit() easily.
                clf.fit(X_train, y_train)

                return clf

        else:
            raise ValueError(f"Unknown model_type provided: {model_type}")

    @staticmethod
    def predict_proba(model, X):
        """
        Unified prediction interface to get probabilities for class 1.

        Args:
            model: The trained model object.
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Probabilities of the positive class.
        """
        if model is None:
            return np.zeros(len(X))

        try:
            # All supported models follow the sklearn predict_proba API
            probs = model.predict_proba(X)

            # Return probability of class 1
            if probs.shape[1] == 2:
                return probs[:, 1]
            else:
                # Handle edge cases (e.g., only one class in training)
                # If only class 0 exists, probs might be shape (N, 1) or (N, 2) depending on implementation
                if probs.shape[1] > 1:
                    return probs[:, 1]
                else:
                    return probs[:, 0]
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return np.zeros(len(X))
