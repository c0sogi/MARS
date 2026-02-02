import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model_factory import ModelFactory
from library.utils import set_seed


class StackingEngine:
    """
    Manages the training, cross-validation, and inference of the
    Multi-Paradigm Stacking Ensemble.
    """

    def __init__(self):
        self.meta_learner = None
        self.base_lexical_rf = None
        self.base_semantic_rf = None
        self.base_semantic_xgb = None

        # Placeholders for retrained models
        self.final_lexical_rf = None
        self.final_semantic_rf = None
        self.final_semantic_xgb = None

    def fit_cv(self, X_lexical, X_semantic, y):
        """
        Performs K-Fold Cross-Validation to generate Out-Of-Fold (OOF) predictions
        and trains the Level 2 Meta-Learner.

        Args:
            X_lexical (scipy.sparse.csr_matrix): Sparse features for Lexical Bagger.
            X_semantic (np.ndarray): Dense features for Semantic models.
            y (np.ndarray): Target labels.
        """
        set_seed()

        n_folds = Config.N_FOLDS
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        # Arrays to store OOF predictions
        n_samples = y.shape[0]
        oof_lex_rf = np.zeros(n_samples)
        oof_sem_rf = np.zeros(n_samples)
        oof_sem_xgb = np.zeros(n_samples)

        print(f"Starting {n_folds}-Fold Cross-Validation Stacking...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_semantic, y)):
            print(f"Processing Fold {fold + 1}/{n_folds}...")

            # Split Data
            # Lexical View
            X_lex_train, X_lex_val = X_lexical[train_idx], X_lexical[val_idx]

            # Semantic View
            X_sem_train, X_sem_val = X_semantic[train_idx], X_semantic[val_idx]

            # Targets
            y_train, y_val = y[train_idx], y[val_idx]

            # --- 1. Lexical Bagger (RF) ---
            model_lex_rf = ModelFactory.create_lexical_bagger()
            model_lex_rf.fit(X_lex_train, y_train)
            # Predict proba for positive class
            oof_lex_rf[val_idx] = model_lex_rf.predict_proba(X_lex_val)[:, 1]

            # --- 2. Semantic Bagger (RF) ---
            model_sem_rf = ModelFactory.create_semantic_bagger()
            model_sem_rf.fit(X_sem_train, y_train)
            oof_sem_rf[val_idx] = model_sem_rf.predict_proba(X_sem_val)[:, 1]

            # --- 3. Semantic Booster (XGB) ---
            model_sem_xgb = ModelFactory.create_semantic_booster()
            # XGBoost requires eval_set for early stopping
            model_sem_xgb.fit(
                X_sem_train, y_train, eval_set=[(X_sem_val, y_val)], verbose=False
            )
            oof_sem_xgb[val_idx] = model_sem_xgb.predict_proba(X_sem_val)[:, 1]

        # --- Evaluate Level 1 Performance ---
        auc_lex_rf = roc_auc_score(y, oof_lex_rf)
        auc_sem_rf = roc_auc_score(y, oof_sem_rf)
        auc_sem_xgb = roc_auc_score(y, oof_sem_xgb)

        print("\nLevel 1 OOF AUC Scores:")
        print(f"Lexical Bagger (RF): {auc_lex_rf}")
        print(f"Semantic Bagger (RF): {auc_sem_rf}")
        print(f"Semantic Booster (XGB): {auc_sem_xgb}")

        # --- Train Level 2 Meta-Learner ---
        print("\nTraining Level 2 Meta-Learner...")

        # Stack OOF predictions to create meta-features
        X_meta = np.column_stack([oof_lex_rf, oof_sem_rf, oof_sem_xgb])

        self.meta_learner = ModelFactory.create_meta_learner()
        self.meta_learner.fit(X_meta, y)

        # Evaluate Meta-Learner on OOF (approximate performance)
        # Note: Strictly speaking, this is training accuracy on OOFs, but serves as a sanity check
        meta_preds = self.meta_learner.predict_proba(X_meta)[:, 1]
        auc_meta = roc_auc_score(y, meta_preds)
        print(f"Meta-Learner OOF AUC Score: {auc_meta}")

        # Print Meta-Learner Coefficients to see contribution of each base model
        coefs = self.meta_learner.coef_[0]
        print(
            f"Meta-Learner Coefficients: Lexical_RF={coefs[0]}, Semantic_RF={coefs[1]}, Semantic_XGB={coefs[2]}"
        )

    def retrain_base_models(self, X_lexical, X_semantic, y):
        """
        Retrains all Level 1 base learners on the full dataset for final inference.

        Args:
            X_lexical (scipy.sparse.csr_matrix): Full sparse features.
            X_semantic (np.ndarray): Full dense features.
            y (np.ndarray): Full target labels.
        """
        set_seed()
        print("\nRetraining Base Models on Full Dataset...")

        # --- 1. Lexical Bagger (RF) ---
        print("Retraining Lexical Bagger...")
        self.final_lexical_rf = ModelFactory.create_lexical_bagger()
        self.final_lexical_rf.fit(X_lexical, y)

        # --- 2. Semantic Bagger (RF) ---
        print("Retraining Semantic Bagger...")
        self.final_semantic_rf = ModelFactory.create_semantic_bagger()
        self.final_semantic_rf.fit(X_semantic, y)

        # --- 3. Semantic Booster (XGB) ---
        print("Retraining Semantic Booster...")
        self.final_semantic_xgb = ModelFactory.create_semantic_booster()

        # For XGBoost, we need a validation set for early stopping to work effectively.
        # We create a small internal split (10%) from the full data just for this purpose.
        # This is safer than training blindly with a fixed number of trees.
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_semantic, y, test_size=0.1, stratify=y, random_state=Config.SEED
        )

        self.final_semantic_xgb.fit(
            X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False
        )

    def predict(self, X_lexical, X_semantic):
        """
        Generates final predictions for the test set using the stacked ensemble.

        Args:
            X_lexical (scipy.sparse.csr_matrix): Test sparse features.
            X_semantic (np.ndarray): Test dense features.

        Returns:
            np.ndarray: Predicted probabilities of success.
        """
        if any(
            m is None
            for m in [
                self.final_lexical_rf,
                self.final_semantic_rf,
                self.final_semantic_xgb,
                self.meta_learner,
            ]
        ):
            raise RuntimeError(
                "Models have not been trained. Call fit_cv and retrain_base_models first."
            )

        print("Generating Level 1 Predictions for Test Set...")

        # Get probabilities from base learners
        pred_lex_rf = self.final_lexical_rf.predict_proba(X_lexical)[:, 1]
        pred_sem_rf = self.final_semantic_rf.predict_proba(X_semantic)[:, 1]
        pred_sem_xgb = self.final_semantic_xgb.predict_proba(X_semantic)[:, 1]

        # Stack predictions
        X_meta_test = np.column_stack([pred_lex_rf, pred_sem_rf, pred_sem_xgb])

        print("Generating Final Level 2 Predictions...")
        final_predictions = self.meta_learner.predict_proba(X_meta_test)[:, 1]

        return final_predictions
