import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, save_pickle, load_pickle


class StackingEnsemble:
    """
    Implements a Manifold-Regularized Stacking Ensemble.
    Level 1: SVR, KNN, ExtraTrees, LightGBM.
    Level 2: Ridge Regression.
    """

    def __init__(self):
        seed_everything(Config.SEED)

        # Initialize Base Learners using Config parameters
        self.svr = SVR(**Config.SVR_PARAMS)
        self.knn = KNeighborsRegressor(**Config.KNN_PARAMS)
        self.et = ExtraTreesRegressor(**Config.ET_PARAMS)
        self.lgbm = lgb.LGBMRegressor(**Config.LGBM_PARAMS)

        # Initialize Meta Learner
        self.meta_learner = Ridge(**Config.META_PARAMS)

        # Internal state
        self.lgbm_best_iter = Config.LGBM_PARAMS.get("n_estimators", 5000)
        self.fitted_models = {}

    def train_cv(self, X, y):
        """
        Performs 5-Fold Cross-Validation to generate Out-of-Fold (OOF) predictions
        and train the Meta-Learner.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.

        Returns:
            float: The RMSE of the ensemble on the OOF predictions.
        """
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        # Matrix to store OOF predictions for 4 base models
        oof_preds = np.zeros((X.shape[0], 4))

        # Metric tracking
        rmse_scores = {"svr": [], "knn": [], "et": [], "lgbm": []}
        lgbm_iters = []

        print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # --- 1. SVR ---
            self.svr.fit(X_train, y_train)
            pred_svr = self.svr.predict(X_val)
            oof_preds[val_idx, 0] = pred_svr
            rmse_scores["svr"].append(np.sqrt(mean_squared_error(y_val, pred_svr)))

            # --- 2. KNN ---
            # Cite debug_lesson_7: Dynamically Scale Dimensionality Parameters to Input Size
            n_neighbors = min(Config.KNN_PARAMS["n_neighbors"], len(X_train))
            self.knn.set_params(n_neighbors=n_neighbors)
            self.knn.fit(X_train, y_train)
            pred_knn = self.knn.predict(X_val)
            oof_preds[val_idx, 1] = pred_knn
            rmse_scores["knn"].append(np.sqrt(mean_squared_error(y_val, pred_knn)))

            # --- 3. ExtraTrees ---
            self.et.fit(X_train, y_train)
            pred_et = self.et.predict(X_val)
            oof_preds[val_idx, 2] = pred_et
            rmse_scores["et"].append(np.sqrt(mean_squared_error(y_val, pred_et)))

            # --- 4. LightGBM ---
            # Configure callbacks for early stopping and silence
            callbacks = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            fit_params = Config.LGBM_FIT_PARAMS.copy()

            self.lgbm.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
                **fit_params,
            )

            # Predict using the best iteration found
            pred_lgbm = self.lgbm.predict(X_val)
            oof_preds[val_idx, 3] = pred_lgbm
            lgbm_iters.append(self.lgbm.best_iteration_)
            rmse_scores["lgbm"].append(np.sqrt(mean_squared_error(y_val, pred_lgbm)))

            print(f"Fold {fold+1} completed.")

        # Calculate average optimal iterations for LightGBM for final training
        self.lgbm_best_iter = int(np.mean(lgbm_iters))
        print(f"Average LightGBM Best Iteration: {self.lgbm_best_iter}")

        # Print CV Metrics with full precision
        for model_name, scores in rmse_scores.items():
            print(f"CV RMSE ({model_name}): {np.mean(scores)}")

        # --- Train Meta-Learner ---
        # The meta-learner is trained on the OOF predictions to learn how to correct/combine them
        print("Training Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y)
        self.fitted_models["meta"] = self.meta_learner

        # Calculate Ensemble OOF Score
        oof_ensemble_pred = self.meta_learner.predict(oof_preds)
        oof_score = np.sqrt(mean_squared_error(y, oof_ensemble_pred))
        print(f"OOF Ensemble RMSE: {oof_score}")

        return oof_score

    def fit_final(self, X, y):
        """
        Retrains all base learners on the full dataset and saves them.
        Uses the average best iteration for LightGBM determined in CV.

        Args:
            X (np.ndarray): Full training features.
            y (np.ndarray): Full training targets.
        """
        print("Retraining Base Learners on Full Dataset...")

        # 1. SVR
        print("Fitting SVR...")
        self.svr.fit(X, y)
        self.fitted_models["svr"] = self.svr
        save_pickle(self.svr, os.path.join(Config.WORKING_DIR, "svr_final.joblib"))

        # 2. KNN
        print("Fitting KNN...")
        # Cite debug_lesson_7: Dynamically Scale Dimensionality Parameters to Input Size
        n_neighbors = min(Config.KNN_PARAMS["n_neighbors"], len(X))
        self.knn.set_params(n_neighbors=n_neighbors)
        self.knn.fit(X, y)
        self.fitted_models["knn"] = self.knn
        save_pickle(self.knn, os.path.join(Config.WORKING_DIR, "knn_final.joblib"))

        # 3. ExtraTrees
        print("Fitting ExtraTrees...")
        self.et.fit(X, y)
        self.fitted_models["et"] = self.et
        save_pickle(self.et, os.path.join(Config.WORKING_DIR, "et_final.joblib"))

        # 4. LightGBM
        print(f"Fitting LightGBM (n_estimators={self.lgbm_best_iter})...")
        self.lgbm.set_params(n_estimators=self.lgbm_best_iter)
        # No early stopping here as we use the full dataset (no valid set)
        self.lgbm.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        self.fitted_models["lgbm"] = self.lgbm
        save_pickle(self.lgbm, os.path.join(Config.WORKING_DIR, "lgbm_final.joblib"))

        # 5. Meta Learner
        # We save the meta-learner trained on OOFs in train_cv.
        # We do NOT retrain it on base predictions of the full train set to avoid overfitting.
        if "meta" in self.fitted_models:
            save_pickle(
                self.fitted_models["meta"],
                os.path.join(Config.WORKING_DIR, "meta_learner.joblib"),
            )
        else:
            print("Warning: Meta-learner was not trained (train_cv not run).")

    def predict(self, X):
        """
        Generates predictions using the stacked ensemble.
        Loads models from disk if not in memory.

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Predicted Pawpularity scores.
        """
        # Ensure models are loaded
        required_models = ["svr", "knn", "et", "lgbm"]

        for m_name in required_models:
            if m_name not in self.fitted_models:
                path = os.path.join(Config.WORKING_DIR, f"{m_name}_final.joblib")
                if os.path.exists(path):
                    self.fitted_models[m_name] = load_pickle(path)
                else:
                    raise ValueError(f"Model {m_name} not found in memory or at {path}")

        if "meta" not in self.fitted_models:
            path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
            if os.path.exists(path):
                self.fitted_models["meta"] = load_pickle(path)
            else:
                raise ValueError("Meta-learner model not found.")

        # Generate Base Predictions
        p_svr = self.fitted_models["svr"].predict(X)
        p_knn = self.fitted_models["knn"].predict(X)
        p_et = self.fitted_models["et"].predict(X)
        p_lgbm = self.fitted_models["lgbm"].predict(X)

        # Stack Predictions
        X_meta = np.column_stack([p_svr, p_knn, p_et, p_lgbm])

        # Generate Final Prediction
        final_pred = self.fitted_models["meta"].predict(X_meta)

        # Clip to valid range [1, 100]
        return np.clip(final_pred, 1.0, 100.0)

    def generate_submission(self, X_test, test_ids):
        """
        Generates predictions for the test set and saves the submission CSV.

        Args:
            X_test (np.ndarray): Test features.
            test_ids (np.ndarray): Test IDs.
        """
        print("Generating submission...")
        preds = self.predict(X_test)

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        df = pd.DataFrame({"Id": test_ids, "Pawpularity": preds})

        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
