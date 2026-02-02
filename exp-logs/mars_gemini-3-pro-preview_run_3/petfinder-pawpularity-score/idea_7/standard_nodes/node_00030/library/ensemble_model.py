import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from library.config import Config
from library.utils import get_logger, seed_everything, calculate_rmse


class EnsembleTrainer:
    """
    Manages the training of the Stacking Ensemble:
    1. Level 1: SVR, LightGBM, ExtraTrees (trained via 5-Fold CV)
    2. Level 2: Linear Regression (Meta-Learner)
    """

    def __init__(self):
        self.config = Config
        self.logger = get_logger("EnsembleTrainer")
        self.working_dir = self.config.WORKING_DIR
        self.models_dir = os.path.join(self.working_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.seed = self.config.SEED
        seed_everything(self.seed)

    def _get_svr(self):
        # SVR requires scaling
        return make_pipeline(StandardScaler(), SVR(**self.config.SVR_PARAMS))

    def _get_lgbm(self):
        return LGBMRegressor(**self.config.LGBM_PARAMS)

    def _get_extratrees(self):
        return ExtraTreesRegressor(**self.config.EXTRATREES_PARAMS)

    def _get_meta_learner(self):
        return LinearRegression(**self.config.META_LEARNER_PARAMS)

    def train_level1_cv(self, X, y):
        """
        Performs K-Fold CV to generate OOF predictions and determine optimal hyperparameters (e.g., n_estimators for LGBM).
        """
        self.logger.info(
            f"Starting Level 1 Cross-Validation (Folds={self.config.N_FOLDS})..."
        )

        kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.seed)

        # Placeholders for OOF predictions
        oof_preds = {
            "svr": np.zeros(len(X)),
            "lgbm": np.zeros(len(X)),
            "extratrees": np.zeros(len(X)),
        }

        lgbm_best_iters = []

        fold_rmses = {k: [] for k in oof_preds.keys()}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # --- SVR ---
            model_svr = self._get_svr()
            model_svr.fit(X_train_fold, y_train_fold)
            pred_svr = model_svr.predict(X_val_fold)
            oof_preds["svr"][val_idx] = pred_svr
            fold_rmses["svr"].append(calculate_rmse(y_val_fold, pred_svr))

            # --- ExtraTrees ---
            model_et = self._get_extratrees()
            model_et.fit(X_train_fold, y_train_fold)
            pred_et = model_et.predict(X_val_fold)
            oof_preds["extratrees"][val_idx] = pred_et
            fold_rmses["extratrees"].append(calculate_rmse(y_val_fold, pred_et))

            # --- LightGBM ---
            model_lgbm = self._get_lgbm()
            # Use early stopping
            callbacks = [
                early_stopping(
                    stopping_rounds=self.config.LGBM_EARLY_STOPPING_ROUNDS,
                    verbose=False,
                ),
                log_evaluation(period=0),  # Silent
            ]
            model_lgbm.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric="rmse",
                callbacks=callbacks,
            )
            pred_lgbm = model_lgbm.predict(X_val_fold)
            oof_preds["lgbm"][val_idx] = pred_lgbm
            fold_rmses["lgbm"].append(calculate_rmse(y_val_fold, pred_lgbm))

            # Record best iteration
            if model_lgbm.best_iteration_:
                lgbm_best_iters.append(model_lgbm.best_iteration_)
            else:
                lgbm_best_iters.append(self.config.LGBM_PARAMS["n_estimators"])

            self.logger.info(f"Fold {fold+1}/{self.config.N_FOLDS} completed.")

        # Log Average RMSE
        self.logger.info("Level 1 CV Results (RMSE):")
        for model_name, scores in fold_rmses.items():
            avg_score = np.mean(scores)
            self.logger.info(f"  {model_name}: {avg_score}")

        avg_best_iter = int(np.mean(lgbm_best_iters))
        self.logger.info(f"Average Best Iteration for LGBM: {avg_best_iter}")

        return oof_preds, avg_best_iter

    def train_meta_learner(self, oof_preds, y):
        """
        Trains the Level 2 Linear Regression on OOF predictions.
        """
        self.logger.info("Training Level 2 Meta-Learner...")

        # Stack OOF predictions: Shape (N_samples, 3)
        X_meta = np.column_stack(
            [oof_preds["svr"], oof_preds["lgbm"], oof_preds["extratrees"]]
        )

        meta_model = self._get_meta_learner()
        meta_model.fit(X_meta, y)

        # Check coefficients
        self.logger.info(
            f"Meta-Learner Coefficients: SVR={meta_model.coef_[0]:.4f}, LGBM={meta_model.coef_[1]:.4f}, ET={meta_model.coef_[2]:.4f}"
        )
        self.logger.info(f"Meta-Learner Intercept: {meta_model.intercept_:.4f}")

        # Calculate Ensemble RMSE on OOF
        final_oof_preds = meta_model.predict(X_meta)
        ensemble_rmse = calculate_rmse(y, final_oof_preds)
        self.logger.info(f"Ensemble OOF RMSE: {ensemble_rmse}")

        return meta_model

    def retrain_level1_full(self, X, y, lgbm_n_estimators):
        """
        Retrains Level 1 models on the full dataset (Train + Val).
        """
        self.logger.info("Retraining Level 1 models on full dataset...")

        # --- SVR ---
        model_svr = self._get_svr()
        model_svr.fit(X, y)

        # --- ExtraTrees ---
        model_et = self._get_extratrees()
        model_et.fit(X, y)

        # --- LightGBM ---
        # Update n_estimators to the average best found in CV
        lgbm_params = self.config.LGBM_PARAMS.copy()
        lgbm_params["n_estimators"] = lgbm_n_estimators
        model_lgbm = LGBMRegressor(**lgbm_params)
        model_lgbm.fit(X, y)

        return {"svr": model_svr, "lgbm": model_lgbm, "extratrees": model_et}

    def predict(self, models_l1, model_meta, X_test):
        """
        Generates final predictions using the ensemble.
        """
        p_svr = models_l1["svr"].predict(X_test)
        p_lgbm = models_l1["lgbm"].predict(X_test)
        p_et = models_l1["extratrees"].predict(X_test)

        X_meta_test = np.column_stack([p_svr, p_lgbm, p_et])
        final_preds = model_meta.predict(X_meta_test)

        # Clip to valid range
        final_preds = np.clip(final_preds, 1.0, 100.0)

        return final_preds

    def run(
        self, X_train, y_train, X_val, y_val, X_test, ids_test, load_cached_data=True
    ):
        """
        Main execution pipeline.
        """
        # Define cache paths
        cache_path = os.path.join(self.models_dir, "ensemble_models.joblib")

        # Combine Train and Val for Stacking
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        models_l1 = None
        model_meta = None

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info("Loading cached ensemble models...")
            try:
                cached_data = joblib.load(cache_path)
                models_l1 = cached_data["level1"]
                model_meta = cached_data["meta"]
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Retraining...")

        if models_l1 is None or model_meta is None:
            # 1. Level 1 CV
            oof_preds, avg_lgbm_iter = self.train_level1_cv(X_full, y_full)

            # 2. Train Meta Learner
            model_meta = self.train_meta_learner(oof_preds, y_full)

            # 3. Final Retraining
            models_l1 = self.retrain_level1_full(X_full, y_full, avg_lgbm_iter)

            # Save to cache
            self.logger.info(f"Saving models to {cache_path}...")
            joblib.dump({"level1": models_l1, "meta": model_meta}, cache_path)

        # 4. Predict on Test Set
        self.logger.info("Generating predictions for Test Set...")
        final_predictions = self.predict(models_l1, model_meta, X_test)

        # 5. Create Submission
        submission_df = pd.DataFrame({"Id": ids_test, "Pawpularity": final_predictions})

        submission_path = self.config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")
        self.logger.info(f"Sample predictions:\n{submission_df.head()}")

        return submission_df
