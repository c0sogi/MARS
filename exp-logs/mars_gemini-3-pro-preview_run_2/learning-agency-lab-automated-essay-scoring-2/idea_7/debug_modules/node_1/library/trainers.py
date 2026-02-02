import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.sparse import vstack
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.feature_engineering import FeatureExtractor
import library.model_nn as model_nn


class OptimizedRounder:
    """
    Nelder-Mead optimization to find optimal thresholds for rounding continuous scores
    to the integer scale 1-6.
    """

    def __init__(self):
        self.coef_ = [1.5, 2.5, 3.5, 4.5, 5.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        # Apply thresholds
        # Note: We assume coef is sorted, but Nelder-Mead doesn't guarantee it.
        # We process in order of standard thresholds to be robust or sort them.
        # For simplicity and speed in the loop, we use the provided coefs directly.
        # A robust implementation sorts them first.
        c = np.sort(coef)

        # Vectorized thresholding
        X_p = pd.cut(
            X_p, [-np.inf] + list(c) + [np.inf], labels=[1, 2, 3, 4, 5, 6]
        ).astype(int)

        ll = compute_qwk(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5]
        # Nelder-Mead is robust for derivative-free optimization
        self.coef_ = minimize(
            loss_partial, initial_coef, method="nelder-mead", tol=1e-6
        ).x
        self.coef_ = np.sort(self.coef_)

    def predict(self, X, coef):
        X_p = np.copy(X)
        c = np.sort(coef)
        X_p = pd.cut(
            X_p, [-np.inf] + list(c) + [np.inf], labels=[1, 2, 3, 4, 5, 6]
        ).astype(int)
        return X_p


class Trainer:
    """
    Orchestrates the training of the Quad-Branch Heterogeneous Stacking Network.
    Manages Semantic (DeBERTa), Lexical (Ridge), Morphological (Ridge), and Meta (LightGBM) branches.
    """

    def __init__(self):
        Config.setup()
        seed_everything(Config.SEED)
        self.fe = FeatureExtractor()

    def _get_data_and_folds(self):
        """
        Loads metadata, concatenates Train and Val to form the full development set,
        and generates stratified folds consistent across all branches.
        """
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)

        # Handle Debug Mode alignment with model_nn
        if Config.DEBUG:
            df_train = df_train.head(50)
            df_val = df_val.head(50)

        # Concatenate to match the CV strategy
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        y = df_full["score"].values

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        folds = list(skf.split(np.zeros(len(y)), y.astype(int)))

        return df_full, y, folds

    def train_semantic_branch(self):
        """
        Executes the Semantic Branch training using the Deep Learning module.
        Wraps library.model_nn.train_semantic_branch.
        """
        print("\n=== Semantic Branch (DeBERTa) ===")
        # Check if predictions already exist to avoid re-training
        oof_path = os.path.join(Config.WORKING_DIR, "train_semantic_preds.parquet")
        if os.path.exists(oof_path):
            print(
                f"Found existing semantic predictions at {oof_path}. Skipping training."
            )
            return

        # Delegate to the NN module which handles the full CV loop, AWP, and saving
        model_nn.train_semantic_branch()

    def train_ridge_branch(self, kind="word"):
        """
        Trains a Ridge Regression branch (Lexical or Morphological).

        Args:
            kind (str): 'word' for Lexical branch, 'char' for Morphological branch.
        """
        print(f"\n=== {kind.capitalize()} Branch (Ridge) ===")

        # Load Features
        train_csr, val_csr, test_csr = self.fe.get_tfidf_features(kind=kind)

        # Handle Debug Mode slicing for features
        if Config.DEBUG:
            train_csr = train_csr[:50]
            val_csr = val_csr[:50]
            test_csr = test_csr[
                :50
            ]  # Assuming test is also subsampled in debug logic if needed

        # Combine Train/Val features
        X_full = vstack([train_csr, val_csr])

        # Get Labels and Folds
        _, y_full, folds = self._get_data_and_folds()

        # Prepare containers
        oof_preds = np.zeros(len(y_full))
        test_preds_list = []

        # Determine Hyperparameters
        alpha = Config.RIDGE_ALPHA_WORD if kind == "word" else Config.RIDGE_ALPHA_CHAR

        # CV Loop
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]

            model = Ridge(alpha=alpha, random_state=Config.SEED, solver="auto")
            model.fit(X_train, y_train)

            # Predict
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred

            test_pred = model.predict(test_csr)
            test_preds_list.append(test_pred)

            # Save Model
            model_path = os.path.join(
                Config.MODEL_OUTPUT_DIR, f"ridge_{kind}_fold_{fold}.joblib"
            )
            joblib.dump(model, model_path)

            mse = mean_squared_error(y_val, val_pred)
            print(f"Fold {fold} MSE: {mse:.5f}")

        # Aggregate Test Predictions
        avg_test_preds = np.mean(test_preds_list, axis=0)

        # Save Predictions
        pd.DataFrame({"pred": oof_preds}).to_parquet(
            os.path.join(Config.WORKING_DIR, f"train_{kind}_preds.parquet")
        )
        pd.DataFrame({"pred": avg_test_preds}).to_parquet(
            os.path.join(Config.WORKING_DIR, f"test_{kind}_preds.parquet")
        )

        qwk = compute_qwk(y_full.astype(int), np.rint(oof_preds).clip(1, 6).astype(int))
        print(f"{kind.capitalize()} Branch OOF QWK: {qwk:.6f}")

    def train_meta_learner(self):
        """
        Trains the Meta-Learner (LightGBM) using Stacking.
        Combines OOF predictions from all branches + Structural features.
        Optimizes thresholds and generates final submission.
        """
        print("\n=== Meta Learner (Stacking) ===")

        # 1. Load Base Predictions (OOF and Test)
        try:
            sem_oof = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "train_semantic_preds.parquet")
            )["pred"].values
            sem_test = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "test_semantic_preds.parquet")
            )["pred"].values

            lex_oof = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "train_word_preds.parquet")
            )["pred"].values
            lex_test = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "test_word_preds.parquet")
            )["pred"].values

            mor_oof = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "train_char_preds.parquet")
            )["pred"].values
            mor_test = pd.read_parquet(
                os.path.join(Config.WORKING_DIR, "test_char_preds.parquet")
            )["pred"].values
        except FileNotFoundError as e:
            print(
                f"Error loading base predictions: {e}. Ensure all branches are trained."
            )
            return

        # 2. Load Structural Features
        s_train, s_val, s_test = self.fe.get_structural_features()

        if Config.DEBUG:
            s_train = s_train.head(50)
            s_val = s_val.head(50)
            s_test = s_test.head(50)  # Assuming consistency

        s_full = pd.concat([s_train, s_val], axis=0).reset_index(drop=True)

        # 3. Construct Meta Feature Matrix
        # Features: [Semantic, Lexical, Morphological, ...Structural...]
        X_meta = np.column_stack([sem_oof, lex_oof, mor_oof, s_full.values])
        X_test_meta = np.column_stack([sem_test, lex_test, mor_test, s_test.values])

        _, y_full, folds = self._get_data_and_folds()

        oof_meta = np.zeros(len(y_full))
        test_meta_preds = []

        # 4. Train LightGBM
        lgbm_params = Config.META_MODEL_PARAMS.copy()
        es_rounds = lgbm_params.pop("early_stopping_rounds", 50)

        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]

            model = lgb.LGBMRegressor(**lgbm_params, random_state=Config.SEED)

            callbacks = [
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=callbacks,
            )

            val_pred = model.predict(X_val)
            oof_meta[val_idx] = val_pred

            test_pred = model.predict(X_test_meta)
            test_meta_preds.append(test_pred)

        avg_test_meta = np.mean(test_meta_preds, axis=0)

        # 5. Optimize Thresholds
        print("Optimizing thresholds using Nelder-Mead...")
        rounder = OptimizedRounder()
        rounder.fit(oof_meta, y_full.astype(int))
        print(f"Optimal Coefficients: {rounder.coef_}")

        final_oof_rounded = rounder.predict(oof_meta, rounder.coef_)
        final_qwk = compute_qwk(y_full.astype(int), final_oof_rounded)
        print(f"Final Ensemble OOF QWK: {final_qwk:.6f}")

        # 6. Generate Submission
        final_test_rounded = rounder.predict(avg_test_meta, rounder.coef_)

        # Load sample submission for IDs
        sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Ensure length matches (handle debug case if necessary, though test usually fixed)
        if len(final_test_rounded) != len(sub):
            print(
                f"Warning: Prediction length {len(final_test_rounded)} != Submission length {len(sub)}"
            )
            # In debug, we might have sliced test, so we slice submission
            sub = sub.iloc[: len(final_test_rounded)]

        sub["score"] = final_test_rounded
        sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    def run(self):
        """
        Executes the full pipeline.
        """
        self.train_semantic_branch()
        self.train_ridge_branch(kind="word")
        self.train_ridge_branch(kind="char")
        self.train_meta_learner()
