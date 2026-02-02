import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib

from library.config import Config
from library.utils import seed_everything, multiclass_log_loss
from library.data_factory import DataManager


class ClassicalModels:
    """
    Manages the training and inference of classical machine learning models
    (Logistic Regression, Naive Bayes, XGBoost) using Stratified K-Fold CV.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.n_folds = Config.N_FOLDS
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_data(self, load_cached_data=True):
        """
        Loads metadata and features, then concatenates train and validation sets
        to form the full cross-validation dataset.
        """
        # Load Metadata
        train_df, val_df, test_df = DataManager.load_metadata()

        # Load Features
        train_tfidf, val_tfidf, test_tfidf = DataManager.get_tfidf_features(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )
        train_svd, val_svd, test_svd = DataManager.get_svd_features(
            train_tfidf, val_tfidf, test_tfidf, load_cached_data=load_cached_data
        )

        # Get Labels
        y_train = train_df[Config.TARGET_COL].map(Config.LABEL_MAP).values
        y_val = val_df[Config.TARGET_COL].map(Config.LABEL_MAP).values

        # Concatenate Train + Val for full CV
        # TF-IDF (Sparse)
        X_tfidf_full = scipy.sparse.vstack([train_tfidf, val_tfidf])
        # SVD (Dense)
        X_svd_full = np.vstack([train_svd, val_svd])
        # Labels
        y_full = np.concatenate([y_train, y_val])

        # Test features remain separate
        return {
            "X_tfidf": X_tfidf_full,
            "X_svd": X_svd_full,
            "y": y_full,
            "X_test_tfidf": test_tfidf,
            "X_test_svd": test_svd,
        }

    def _run_model_cv(self, model_name, model, X, y, X_test, load_cached_data):
        """
        Generic function to run Stratified K-Fold CV for a given model.
        Handles caching of OOF and Test predictions.
        """
        oof_path = os.path.join(self.working_dir, f"oof_{model_name}.npy")
        pred_path = os.path.join(self.working_dir, f"pred_test_{model_name}.npy")

        if load_cached_data and os.path.exists(oof_path) and os.path.exists(pred_path):
            print(f"[{model_name}] Loading cached predictions...")
            oof_preds = np.load(oof_path)
            test_preds = np.load(pred_path)
            return oof_preds, test_preds

        print(f"[{model_name}] Starting {self.n_folds}-Fold CV...")

        # Initialize containers
        oof_preds = np.zeros((len(y), Config.NUM_CLASSES))
        test_preds = np.zeros((X_test.shape[0], Config.NUM_CLASSES))

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # Fit model
            # Special handling for XGBoost to use early stopping
            if model_name == "xgb":
                model.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False,
                )
            else:
                model.fit(X_train_fold, y_train_fold)

            # Predict OOF
            val_probs = model.predict_proba(X_val_fold)
            oof_preds[val_idx] = val_probs

            # Predict Test
            test_fold_probs = model.predict_proba(X_test)
            test_preds += test_fold_probs / self.n_folds

            # Score
            fold_loss = multiclass_log_loss(y_val_fold, val_probs)
            scores.append(fold_loss)
            # print(f"  Fold {fold+1} LogLoss: {fold_loss}") # Optional verbosity

        avg_loss = np.mean(scores)
        print(f"[{model_name}] CV LogLoss: {avg_loss:.15f}")

        # Cache results
        np.save(oof_path, oof_preds)
        np.save(pred_path, test_preds)

        return oof_preds, test_preds

    def run_classical_cv(self, load_cached_data=True):
        """
        Main entry point to run all classical models.
        """
        data = self._get_data(load_cached_data)

        results = {}

        # 1. Logistic Regression (TF-IDF)
        lr_model = LogisticRegression(
            C=1.0,
            solver="saga",
            multi_class="multinomial",
            max_iter=1000,
            random_state=Config.SEED,
            n_jobs=-1,
        )
        oof_lr, pred_lr = self._run_model_cv(
            "lr",
            lr_model,
            data["X_tfidf"],
            data["y"],
            data["X_test_tfidf"],
            load_cached_data,
        )
        results["lr"] = {"oof": oof_lr, "test": pred_lr}

        # 2. Naive Bayes (TF-IDF)
        # NB is very fast and works well with sparse counts/tf-idf
        nb_model = MultinomialNB(alpha=0.01)
        oof_nb, pred_nb = self._run_model_cv(
            "nb",
            nb_model,
            data["X_tfidf"],
            data["y"],
            data["X_test_tfidf"],
            load_cached_data,
        )
        results["nb"] = {"oof": oof_nb, "test": pred_nb}

        # 3. XGBoost (SVD)
        # XGBoost needs dense features, so we use SVD
        xgb_model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            random_state=Config.SEED,
            n_jobs=4,  # Limit threads to avoid contention
            verbosity=0,
        )
        oof_xgb, pred_xgb = self._run_model_cv(
            "xgb",
            xgb_model,
            data["X_svd"],
            data["y"],
            data["X_test_svd"],
            load_cached_data,
        )
        results["xgb"] = {"oof": oof_xgb, "test": pred_xgb}

        return results
