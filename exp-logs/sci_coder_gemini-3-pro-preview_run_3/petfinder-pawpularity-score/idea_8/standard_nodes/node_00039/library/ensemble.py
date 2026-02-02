import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from library.config import Config
from library.utils import seed_everything, save_cache, load_cache


class StackingEnsemble:
    """
    Implements a Stacking Ensemble with SVR, ExtraTrees, and LightGBM as base learners,
    and Linear Regression as the meta-learner.
    """

    def __init__(self):
        self.seed = Config.SEED
        seed_everything(self.seed)

        self.n_folds = Config.N_FOLDS
        self.working_dir = Config.WORKING_DIR
        self.submission_dir = Config.SUBMISSION_DIR

        # Initialize Base Learners
        self.svr_params = Config.SVR_PARAMS
        self.et_params = Config.EXTRATREES_PARAMS
        self.lgbm_params = Config.LGBM_PARAMS

        # Initialize Meta Learner
        self.meta_learner = LinearRegression()

        # Placeholders for trained base models (final retraining)
        self.final_svr = None
        self.final_et = None
        self.final_lgbm = None

        # Track best iterations for LGBM
        self.best_iterations = []

    def _get_base_models(self):
        """Re-instantiates fresh base models."""
        svr = SVR(**self.svr_params)
        et = ExtraTreesRegressor(**self.et_params)
        lgbm = LGBMRegressor(**self.lgbm_params)
        return svr, et, lgbm

    def cross_validate(self, X, y):
        """
        Performs K-Fold Cross Validation to generate OOF predictions.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            oof_preds (np.ndarray): Matrix of OOF predictions from base models (N, 3).
        """
        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # Storage for OOF predictions: (N_samples, N_models)
        # Models: 0=SVR, 1=ET, 2=LGBM
        oof_preds = np.zeros((X.shape[0], 3))

        # Metrics
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Get fresh models
            svr, et, lgbm = self._get_base_models()

            # --- Train SVR ---
            svr.fit(X_tr, y_tr)
            p_svr = svr.predict(X_va)
            oof_preds[val_idx, 0] = p_svr

            # --- Train ExtraTrees ---
            et.fit(X_tr, y_tr)
            p_et = et.predict(X_va)
            oof_preds[val_idx, 1] = p_et

            # --- Train LightGBM ---
            # Use early stopping
            callbacks = [
                early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                ),
                log_evaluation(period=0),  # Silent
            ]
            lgbm.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=callbacks,
            )
            p_lgbm = lgbm.predict(X_va)
            oof_preds[val_idx, 2] = p_lgbm

            # Store best iteration
            if lgbm.best_iteration_:
                self.best_iterations.append(lgbm.best_iteration_)

            # --- Evaluate Fold ---
            # Simple average for fold scoring check (Meta learner will do better)
            p_avg = (p_svr + p_et + p_lgbm) / 3.0
            rmse = np.sqrt(mean_squared_error(y_va, p_avg))
            fold_scores.append(rmse)

            print(
                f"Fold {fold+1}/{self.n_folds} - SVR RMSE: {np.sqrt(mean_squared_error(y_va, p_svr))}"
            )
            print(
                f"Fold {fold+1}/{self.n_folds} - ET RMSE: {np.sqrt(mean_squared_error(y_va, p_et))}"
            )
            print(
                f"Fold {fold+1}/{self.n_folds} - LGBM RMSE: {np.sqrt(mean_squared_error(y_va, p_lgbm))}"
            )
            print(f"Fold {fold+1}/{self.n_folds} - Mean Ensemble RMSE: {rmse}")

        print(f"Average CV RMSE (Mean Ensemble): {np.mean(fold_scores)}")
        return oof_preds

    def fit_final(self, X, y, oof_preds):
        """
        Retrains base models on the full dataset and trains the meta-learner on OOF predictions.

        Args:
            X (np.ndarray): Full feature matrix.
            y (np.ndarray): Full target vector.
            oof_preds (np.ndarray): OOF predictions from CV (N, 3).
        """
        print("Retraining base models on full dataset...")

        self.final_svr, self.final_et, self.final_lgbm = self._get_base_models()

        # SVR
        self.final_svr.fit(X, y)

        # ExtraTrees
        self.final_et.fit(X, y)

        # LightGBM
        # Determine n_estimators from CV
        if self.best_iterations:
            avg_iter = int(np.mean(self.best_iterations))
            print(f"Retraining LGBM with n_estimators={avg_iter} (derived from CV)")
            self.final_lgbm.set_params(n_estimators=avg_iter)

        self.final_lgbm.fit(X, y)

        print("Training Meta-Learner (Linear Regression) on OOF predictions...")
        self.meta_learner.fit(oof_preds, y)

        # Print Meta-Learner Coefficients
        coefs = self.meta_learner.coef_
        intercept = self.meta_learner.intercept_
        print(
            f"Meta-Learner Coefficients: SVR={coefs[0]}, ET={coefs[1]}, LGBM={coefs[2]}"
        )
        print(f"Meta-Learner Intercept: {intercept}")

        # Calculate OOF Score for Meta Learner
        meta_oof_preds = self.meta_learner.predict(oof_preds)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_oof_preds))
        print(f"Meta-Learner OOF RMSE: {meta_rmse}")

        # Save models
        joblib.dump(self.final_svr, os.path.join(self.working_dir, "svr_final.joblib"))
        joblib.dump(self.final_et, os.path.join(self.working_dir, "et_final.joblib"))
        joblib.dump(
            self.final_lgbm, os.path.join(self.working_dir, "lgbm_final.joblib")
        )
        joblib.dump(
            self.meta_learner, os.path.join(self.working_dir, "meta_final.joblib")
        )

    def predict_and_submit(self, X_test, test_ids):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            X_test (np.ndarray): Test features.
            test_ids (np.ndarray): Test IDs.
        """
        print("Generating predictions for test set...")

        if self.final_svr is None:
            raise RuntimeError("Models not trained. Call fit_final first.")

        # Base Predictions
        p_svr = self.final_svr.predict(X_test)
        p_et = self.final_et.predict(X_test)
        p_lgbm = self.final_lgbm.predict(X_test)

        # Stack
        base_preds = np.column_stack([p_svr, p_et, p_lgbm])

        # Meta Prediction
        final_preds = self.meta_learner.predict(base_preds)

        # Clip to valid range
        final_preds = np.clip(final_preds, 1.0, 100.0)  # Pawpularity range

        # Create DataFrame
        df_sub = pd.DataFrame({"Id": test_ids, "Pawpularity": final_preds})

        # Save
        sub_path = os.path.join(self.submission_dir, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
        print(df_sub.head())

    def run(self, X_train, y_train, X_val, y_val, X_test, test_ids):
        """
        Orchestrates the full ensemble pipeline.
        Combines Train and Val for CV and Final Training as per instructions.
        """
        # Combine Train and Val for maximum data usage
        print("Combining Train and Validation sets for Ensemble CV...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # Check for cached OOF predictions
        oof_cache_path = "oof_preds.npy"
        oof_preds = load_cache(oof_cache_path)

        if oof_preds is None:
            oof_preds = self.cross_validate(X_full, y_full)
            save_cache(oof_preds, oof_cache_path)
        else:
            print("Loaded OOF predictions from cache.")

        # Final Fit
        self.fit_final(X_full, y_full, oof_preds)

        # Predict
        self.predict_and_submit(X_test, test_ids)
