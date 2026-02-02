import os
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
import warnings

# Handle optional CatBoost dependency
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from library.config import (
    LGBM_PARAMS,
    XGB_PARAMS,
    CATBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS,
    SEED,
    IDEA_DIR,
)
from library.utils import setup_logger

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")


class LGBMExpert:
    """
    LightGBM Expert Model (Leaf-wise growth).
    Optimized for dense numerical data and high capacity.
    """

    def __init__(self):
        self.logger = setup_logger("lgbm_expert")
        self.params = LGBM_PARAMS.copy()
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Initializing LightGBM Expert...")

        # Initialize classifier
        self.model = lgb.LGBMClassifier(**self.params)

        # Setup callbacks for early stopping
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            )
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.logger.info(f"Training LightGBM on {len(X_train)} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )

        # Log validation score if available
        if eval_set:
            score = self.model.best_score_["valid_0"]["binary_logloss"]
            self.logger.info(f"LightGBM Best Validation LogLoss: {score:.6f}")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("LGBM model not trained.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)


class XGBExpert:
    """
    XGBoost Expert Model (Level-wise growth / Histogram).
    Optimized for approximate splits and high capacity.
    """

    def __init__(self):
        self.logger = setup_logger("xgb_expert")
        self.params = XGB_PARAMS.copy()
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Initializing XGBoost Expert...")

        # Dynamic Scale Pos Weight Calculation if not in params
        if "scale_pos_weight" not in self.params:
            num_pos = np.sum(y_train)
            num_neg = len(y_train) - num_pos
            scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
            self.params["scale_pos_weight"] = scale_weight
            self.logger.info(f"Dynamic scale_pos_weight: {scale_weight:.4f}")

        # Determine early stopping configuration
        early_stopping_rounds = None
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = EARLY_STOPPING_ROUNDS

        # XGBoost >= 1.6.0 requires early_stopping_rounds in constructor
        self.model = xgb.XGBClassifier(
            **self.params, early_stopping_rounds=early_stopping_rounds
        )

        self.logger.info(f"Training XGBoost on {len(X_train)} samples...")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        if eval_set and hasattr(self.model, "best_score"):
            self.logger.info(
                f"XGBoost Best Validation LogLoss: {self.model.best_score:.6f}"
            )

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("XGB model not trained.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)


class CatBoostExpert:
    """
    CatBoost Expert Model (Oblivious/Symmetric trees).
    Optimized for structural diversity and native imbalance handling.
    """

    def __init__(self):
        self.logger = setup_logger("catboost_expert")
        self.params = CATBOOST_PARAMS.copy()
        self.model = None
        if not CATBOOST_AVAILABLE:
            self.logger.warning("CatBoost not installed. This expert will be disabled.")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if not CATBOOST_AVAILABLE:
            return

        self.logger.info("Initializing CatBoost Expert...")
        self.model = CatBoostClassifier(**self.params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.logger.info(f"Training CatBoost on {len(X_train)} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if eval_set else None,
            verbose=False,
        )

        if eval_set:
            self.logger.info(
                f"CatBoost Best Validation LogLoss: {self.model.get_best_score()['validation']['Logloss']:.6f}"
            )

    def predict_proba(self, X):
        if not CATBOOST_AVAILABLE or self.model is None:
            # Return zeros if not available/trained, will be handled by ensemble averaging
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        if self.model is not None:
            # CatBoost has its own saving, but joblib works for the wrapper object usually.
            # However, saving the internal model is safer for CatBoost.
            # We will use joblib on the classifier object which is standard for sklearn-API wrappers.
            joblib.dump(self.model, filepath)

    def load(self, filepath):
        if CATBOOST_AVAILABLE:
            self.model = joblib.load(filepath)


class EnsemblePredictor:
    """
    Unified Heterogeneous Tri-Ensemble Predictor.
    Manages LGBM, XGB, and CatBoost experts and aggregates predictions.
    """

    def __init__(self, model_dir=os.path.join(IDEA_DIR, "models")):
        self.logger = setup_logger("ensemble_predictor")
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.lgbm = LGBMExpert()
        self.xgb = XGBExpert()
        self.cat = CatBoostExpert()

        self.experts = {"lgbm": self.lgbm, "xgb": self.xgb, "cat": self.cat}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains all experts in the ensemble.
        """
        self.logger.info("Starting Ensemble Training...")

        # Train LGBM
        self.lgbm.fit(X_train, y_train, X_val, y_val)
        self.save_expert("lgbm")

        # Train XGB
        self.xgb.fit(X_train, y_train, X_val, y_val)
        self.save_expert("xgb")

        # Train CatBoost (if available)
        if CATBOOST_AVAILABLE:
            self.cat.fit(X_train, y_train, X_val, y_val)
            self.save_expert("cat")
        else:
            self.logger.info("Skipping CatBoost training (not available).")

        self.logger.info("Ensemble Training Complete.")

    def predict_proba(self, X):
        """
        Generates averaged probabilities from all trained experts.
        """
        preds = []

        # LGBM
        if self.lgbm.model is not None:
            p_lgbm = self.lgbm.predict_proba(X)
            preds.append(p_lgbm)

        # XGB
        if self.xgb.model is not None:
            p_xgb = self.xgb.predict_proba(X)
            preds.append(p_xgb)

        # CatBoost
        if CATBOOST_AVAILABLE and self.cat.model is not None:
            p_cat = self.cat.predict_proba(X)
            preds.append(p_cat)

        if not preds:
            raise ValueError("No models available for prediction.")

        # Unweighted Average
        avg_preds = np.mean(preds, axis=0)
        return avg_preds

    def save_expert(self, name):
        """Saves a specific expert to disk."""
        if name not in self.experts:
            return
        path = os.path.join(self.model_dir, f"expert_{name}.joblib")
        self.experts[name].save(path)
        self.logger.info(f"Saved {name} expert to {path}")

    def load_experts(self):
        """Loads all available experts from disk."""
        self.logger.info(f"Loading experts from {self.model_dir}...")

        path_lgbm = os.path.join(self.model_dir, "expert_lgbm.joblib")
        if os.path.exists(path_lgbm):
            self.lgbm.load(path_lgbm)
            self.logger.info("Loaded LGBM expert.")

        path_xgb = os.path.join(self.model_dir, "expert_xgb.joblib")
        if os.path.exists(path_xgb):
            self.xgb.load(path_xgb)
            self.logger.info("Loaded XGB expert.")

        path_cat = os.path.join(self.model_dir, "expert_cat.joblib")
        if os.path.exists(path_cat) and CATBOOST_AVAILABLE:
            self.cat.load(path_cat)
            self.logger.info("Loaded CatBoost expert.")
