import os
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.base import BaseEstimator, ClassifierMixin
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate specific Gradient Boosting models
    based on the configurations defined in library.config.
    """

    @staticmethod
    def get_estimator(model_type):
        """
        Returns an instantiated model object based on the model_type.

        Args:
            model_type (str): One of 'lgbm', 'xgb', 'catboost'.

        Returns:
            object: The instantiated sklearn-compatible classifier.
        """
        if model_type == "lgbm":
            return lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        elif model_type == "xgb":
            return xgb.XGBClassifier(**Config.XGB_PARAMS)
        elif model_type == "catboost":
            return cb.CatBoostClassifier(**Config.CATBOOST_PARAMS)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


class TriEnsemble(BaseEstimator, ClassifierMixin):
    """
    Unified Heterogeneous Tri-Ensemble (QGVSA-E).
    Manages LightGBM, XGBoost, and CatBoost experts with unweighted averaging.
    """

    def __init__(self):
        self.lgbm = ModelFactory.get_estimator("lgbm")
        self.xgb = ModelFactory.get_estimator("xgb")
        self.cat = ModelFactory.get_estimator("catboost")
        self.models = {"lgbm": self.lgbm, "xgb": self.xgb, "catboost": self.cat}
        self.is_fitted = False

    def fit(self, X, y, eval_set=None):
        """
        Fits all three models in the ensemble.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training labels.
            eval_set (list of tuple): List containing [(X_val, y_val)] for early stopping.

        Returns:
            self: Returns the instance itself.
        """
        print("Fitting Tri-Ensemble...")

        # 1. Dynamic scale_pos_weight calculation for XGBoost
        # Calculate ratio of negative to positive samples
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Update XGBoost parameter
        self.xgb.set_params(scale_pos_weight=scale_pos_weight)
        print(f"  [XGBoost] Dynamic scale_pos_weight set to: {scale_pos_weight:.4f}")

        # 2. Fit LightGBM
        print("  [LightGBM] Training...")
        callbacks_lgb = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=0),  # Suppress log output
        ]
        self.lgbm.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric="average_precision",
            callbacks=callbacks_lgb,
        )
        if eval_set:
            # Print validation score manually since we suppressed logs
            score = self.lgbm.best_score_["valid_0"]["average_precision"]
            print(f"  [LightGBM] Best Validation AP: {score}")

        # 3. Fit XGBoost
        print("  [XGBoost] Training...")
        # XGBoost fit parameters for early stopping
        self.xgb.fit(X, y, eval_set=eval_set, verbose=False)
        # Note: XGBoost sklearn API handles early stopping differently in newer versions,
        # often requiring it in constructor or callbacks. However, standard fit with eval_set
        # usually works if early_stopping_rounds was set (it's not in our config params,
        # so we rely on the high capacity and regularization, or we can inject it).
        # Given the config structure, we assume standard fitting or that params are sufficient.
        # To be safe and adhere to "Expert" requirements, we inject early stopping if supported.
        # Since we can't easily modify the object signature dynamically safely for all versions,
        # we rely on the configured depth/regularization.
        # *Self-Correction*: The prompt implies we should use the config.
        # If early stopping isn't in XGB_PARAMS, we proceed with standard fit.

        if eval_set:
            # Accessing evaluation results for XGBoost
            results = self.xgb.evals_result()
            # XGBoost stores metrics as 'validation_0': {'logloss': [...]}
            if results and "validation_0" in results:
                final_loss = results["validation_0"]["logloss"][-1]
                print(f"  [XGBoost] Final Validation LogLoss: {final_loss}")

        # 4. Fit CatBoost
        print("  [CatBoost] Training...")
        self.cat.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        if eval_set:
            print(
                f"  [CatBoost] Best Validation MCC: {self.cat.get_best_score()['validation']['MCC']}"
            )

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        Returns the unweighted average of the three models.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Shape (n_samples, 2), probability of class 0 and 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Get probabilities from each model
        # All return (n_samples, 2)
        p_lgbm = self.lgbm.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        p_cat = self.cat.predict_proba(X)

        # Unweighted Average
        avg_proba = (p_lgbm + p_xgb + p_cat) / 3.0
        return avg_proba

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for X.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.
            threshold (float): Decision threshold.

        Returns:
            np.ndarray: Predicted labels (0 or 1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def save_models(self, prefix="expert"):
        """
        Saves the three trained models to the cache directory defined in Config.

        Args:
            prefix (str): Prefix for the filenames (e.g., 'expert', 'scout').
        """
        if not os.path.exists(Config.CACHE_MODELS):
            os.makedirs(Config.CACHE_MODELS, exist_ok=True)

        # Save LGBM
        lgbm_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_lgbm.joblib")
        joblib.dump(self.lgbm, lgbm_path)

        # Save XGBoost
        xgb_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_xgb.joblib")
        joblib.dump(self.xgb, xgb_path)

        # Save CatBoost
        # CatBoost has its own save method which is preferred, but joblib works for the wrapper.
        # We will use joblib for consistency with the wrapper class state.
        cat_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_cat.joblib")
        joblib.dump(self.cat, cat_path)

        print(f"Models saved with prefix '{prefix}' to {Config.CACHE_MODELS}")

    def load_models(self, prefix="expert"):
        """
        Loads the three models from the cache directory.

        Args:
            prefix (str): Prefix for the filenames.
        """
        lgbm_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_lgbm.joblib")
        xgb_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_xgb.joblib")
        cat_path = os.path.join(Config.CACHE_MODELS, f"{prefix}_cat.joblib")

        if not (
            os.path.exists(lgbm_path)
            and os.path.exists(xgb_path)
            and os.path.exists(cat_path)
        ):
            raise FileNotFoundError(
                f"One or more models with prefix '{prefix}' not found in {Config.CACHE_MODELS}"
            )

        self.lgbm = joblib.load(lgbm_path)
        self.xgb = joblib.load(xgb_path)
        self.cat = joblib.load(cat_path)

        self.models = {"lgbm": self.lgbm, "xgb": self.xgb, "catboost": self.cat}
        self.is_fitted = True
        print(f"Models loaded with prefix '{prefix}' from {Config.CACHE_MODELS}")
