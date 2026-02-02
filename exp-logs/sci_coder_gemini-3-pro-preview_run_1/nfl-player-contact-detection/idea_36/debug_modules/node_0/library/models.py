import os
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from library import config, utils

# Ensure model directory exists
os.makedirs(config.MODEL_DIR, exist_ok=True)


class LGBMExpert:
    """
    Wrapper for LightGBM model implementing the Expert configuration.
    Uses leaf-wise growth and handles class imbalance via 'is_unbalance'.
    """

    def __init__(self):
        self.params = config.LGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with early stopping.
        """
        # Set feature names if available (pandas dataframe)
        if hasattr(X_train, "columns"):
            self.feature_names = X_train.columns.tolist()

        # Initialize classifier
        self.model = lgb.LGBMClassifier(n_estimators=config.N_ESTIMATORS, **self.params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        print(f"Training LightGBM Expert...")

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=config.VERBOSE_EVAL),
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )

        if eval_set:
            # Print full precision metric
            score = self.model.best_score_["valid_0"]["binary_logloss"]
            print(f"LGBM Best Validation LogLoss: {score}")

    def predict(self, X):
        """
        Returns probability of the positive class (contact).
        """
        if self.model is None:
            raise ValueError("LGBM model not trained or loaded.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        joblib.dump(self.model, filepath)
        # Also save feature names to ensure alignment during inference if needed
        if self.feature_names:
            joblib.dump(self.feature_names, filepath + "_features.joblib")

    def load(self, filepath):
        self.model = joblib.load(filepath)
        feat_path = filepath + "_features.joblib"
        if os.path.exists(feat_path):
            self.feature_names = joblib.load(feat_path)


class XGBExpert:
    """
    Wrapper for XGBoost model implementing the Expert configuration.
    Uses level-wise growth (hist) and dynamic scale_pos_weight.
    """

    def __init__(self):
        self.params = config.XGB_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model with early stopping and dynamic balancing.
        """
        if hasattr(X_train, "columns"):
            self.feature_names = X_train.columns.tolist()

        # Calculate scale_pos_weight dynamically
        # sum(negative) / sum(positive)
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        scale_weight = n_neg / (n_pos + 1e-6)  # prevent div by zero

        # Update params
        train_params = self.params.copy()
        train_params["scale_pos_weight"] = scale_weight

        print(f"Training XGBoost Expert (scale_pos_weight={scale_weight:.4f})...")

        self.model = xgb.XGBClassifier(
            n_estimators=config.N_ESTIMATORS,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            **train_params,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=config.VERBOSE_EVAL)

        if eval_set:
            # Retrieve best score. XGBoost stores evaluation results in evals_result() if needed,
            # but best_score is accessible if early stopping is used.
            # However, the sklearn API wrapper makes best_score attribute available via best_score property is not standard,
            # usually it is model.best_score for raw booster.
            # For sklearn wrapper, we can look at eval_results or just rely on the log.
            # We will try to access the evaluation result manually for printing.
            results = self.model.evals_result()
            # Assuming 'validation_0' and 'logloss'
            if results and "validation_0" in results:
                final_loss = results["validation_0"]["logloss"][-1]
                # If early stopping triggered, best_iteration is set
                if hasattr(
                    self.model, "best_iteration"
                ) and self.model.best_iteration < len(
                    results["validation_0"]["logloss"]
                ):
                    final_loss = results["validation_0"]["logloss"][
                        self.model.best_iteration
                    ]
                print(f"XGB Best Validation LogLoss: {final_loss}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("XGB model not trained or loaded.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        joblib.dump(self.model, filepath)
        if self.feature_names:
            joblib.dump(self.feature_names, filepath + "_features.joblib")

    def load(self, filepath):
        self.model = joblib.load(filepath)
        feat_path = filepath + "_features.joblib"
        if os.path.exists(feat_path):
            self.feature_names = joblib.load(feat_path)


class DualEnsemble:
    """
    Unified Heterogeneous Dual-Ensemble.
    Manages both LightGBM and XGBoost experts, handles training, saving/loading,
    and ensemble averaging for predictions.
    """

    def __init__(self):
        self.lgbm = LGBMExpert()
        self.xgb = XGBExpert()
        self.lgbm_path = os.path.join(config.MODEL_DIR, "expert_lgbm.joblib")
        self.xgb_path = os.path.join(config.MODEL_DIR, "expert_xgb.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains both models sequentially.
        """
        print("\n=== Fitting Dual Ensemble ===")

        # Train LightGBM
        self.lgbm.fit(X_train, y_train, X_val, y_val)
        self.lgbm.save(self.lgbm_path)

        # Train XGBoost
        self.xgb.fit(X_train, y_train, X_val, y_val)
        self.xgb.save(self.xgb_path)

        print("Dual Ensemble training complete.")

    def predict(self, X):
        """
        Generates predictions from both models and returns the unweighted average.
        """
        print("Generating ensemble predictions...")
        pred_lgbm = self.lgbm.predict(X)
        pred_xgb = self.xgb.predict(X)

        # Unweighted Average
        ensemble_pred = (pred_lgbm + pred_xgb) / 2.0
        return ensemble_pred

    def load(self):
        """
        Loads both trained models from disk.
        """
        print(f"Loading Dual Ensemble from {config.MODEL_DIR}...")
        if os.path.exists(self.lgbm_path):
            self.lgbm.load(self.lgbm_path)
        else:
            print(f"Warning: LightGBM model not found at {self.lgbm_path}")

        if os.path.exists(self.xgb_path):
            self.xgb.load(self.xgb_path)
        else:
            print(f"Warning: XGBoost model not found at {self.xgb_path}")

    def get_feature_importance(self):
        """
        Returns a dictionary containing feature importances from both models if available.
        """
        importances = {}

        # LightGBM Importance
        if self.lgbm.model is not None:
            try:
                lgbm_imp = self.lgbm.model.feature_importances_
                lgbm_names = (
                    self.lgbm.feature_names
                    if self.lgbm.feature_names
                    else [f"f{i}" for i in range(len(lgbm_imp))]
                )
                importances["lgbm"] = dict(zip(lgbm_names, lgbm_imp))
            except Exception as e:
                print(f"Could not extract LGBM importance: {e}")

        # XGBoost Importance
        if self.xgb.model is not None:
            try:
                xgb_imp = self.xgb.model.feature_importances_
                xgb_names = (
                    self.xgb.feature_names
                    if self.xgb.feature_names
                    else [f"f{i}" for i in range(len(xgb_imp))]
                )
                importances["xgb"] = dict(zip(xgb_names, xgb_imp))
            except Exception as e:
                print(f"Could not extract XGB importance: {e}")

        return importances
