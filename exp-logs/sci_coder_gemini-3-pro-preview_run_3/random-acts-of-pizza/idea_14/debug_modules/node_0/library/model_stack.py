import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class QuadStackingClassifier:
    """
    Quad-View Topology-Matched Stacking Ensemble.

    Level 1 Base Learners:
    1. Lexical Bagger (RandomForest) -> Sparse Text View
    2. Behavioral Bagger (RandomForest) -> Sparse History View
    3. Semantic Booster (XGBoost) -> Dense Embedding View
    4. Contextual Baseline (Logistic Regression) -> Dense Metadata View

    Level 2 Meta Learner:
    - Logistic Regression
    """

    def __init__(self):
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_SEED

        # Initialize Base Learners
        self.lexical_model = RandomForestClassifier(**Config.RF_PARAMS)
        self.behavioral_model = RandomForestClassifier(**Config.RF_PARAMS)

        # Handle XGB params - separate fit params from init params
        # We copy to avoid modifying the global config dict
        self.xgb_params = Config.XGB_PARAMS.copy()
        self.early_stopping_rounds = self.xgb_params.pop("early_stopping_rounds", None)
        self.semantic_model = XGBClassifier(**self.xgb_params)

        self.contextual_model = LogisticRegression(**Config.LOGREG_PARAMS)

        # Initialize Meta Learner
        self.meta_model = LogisticRegression(**Config.META_LEARNER_PARAMS)

        # State flag
        self.models_fitted = False

    def fit(self, X_dict, y):
        """
        Performs 5-Fold CV to generate OOF predictions, trains the meta-learner,
        and then retrains base learners on full data.

        Args:
            X_dict (dict): Dictionary containing 'lexical', 'behavioral', 'semantic', 'contextual' feature matrices.
            y (array-like): Target labels.
        """
        # Prepare OOF arrays
        n_samples = len(y)
        oof_preds = np.zeros((n_samples, 4))  # 4 base learners

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        print(f"Starting {self.n_folds}-Fold Cross-Validation Stacking...")

        fold_aucs = {"lexical": [], "behavioral": [], "semantic": [], "contextual": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            # Split Data for this fold
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # --- 1. Lexical View (RF) ---
            # Sparse matrix slicing
            X_lex_train = X_dict["lexical"][train_idx]
            X_lex_val = X_dict["lexical"][val_idx]

            lex_model = RandomForestClassifier(**Config.RF_PARAMS)
            lex_model.fit(X_lex_train, y_train_fold)
            p_lex = lex_model.predict_proba(X_lex_val)[:, 1]
            oof_preds[val_idx, 0] = p_lex
            fold_aucs["lexical"].append(roc_auc_score(y_val_fold, p_lex))

            # --- 2. Behavioral View (RF) ---
            # Sparse matrix slicing
            X_beh_train = X_dict["behavioral"][train_idx]
            X_beh_val = X_dict["behavioral"][val_idx]

            beh_model = RandomForestClassifier(**Config.RF_PARAMS)
            beh_model.fit(X_beh_train, y_train_fold)
            p_beh = beh_model.predict_proba(X_beh_val)[:, 1]
            oof_preds[val_idx, 1] = p_beh
            fold_aucs["behavioral"].append(roc_auc_score(y_val_fold, p_beh))

            # --- 3. Semantic View (XGB) ---
            # Dense matrix slicing
            X_sem_train = X_dict["semantic"][train_idx]
            X_sem_val = X_dict["semantic"][val_idx]

            sem_model = XGBClassifier(**self.xgb_params)
            # Use early stopping if configured
            if self.early_stopping_rounds:
                sem_model.fit(
                    X_sem_train,
                    y_train_fold,
                    eval_set=[(X_sem_val, y_val_fold)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            else:
                sem_model.fit(X_sem_train, y_train_fold)

            p_sem = sem_model.predict_proba(X_sem_val)[:, 1]
            oof_preds[val_idx, 2] = p_sem
            fold_aucs["semantic"].append(roc_auc_score(y_val_fold, p_sem))

            # --- 4. Contextual View (LogReg) ---
            # Dense matrix slicing
            X_ctx_train = X_dict["contextual"][train_idx]
            X_ctx_val = X_dict["contextual"][val_idx]

            ctx_model = LogisticRegression(**Config.LOGREG_PARAMS)
            ctx_model.fit(X_ctx_train, y_train_fold)
            p_ctx = ctx_model.predict_proba(X_ctx_val)[:, 1]
            oof_preds[val_idx, 3] = p_ctx
            fold_aucs["contextual"].append(roc_auc_score(y_val_fold, p_ctx))

            print(f"Fold {fold+1} processed.")

        # Print CV Metrics
        print("\n--- Cross-Validation Results (AUC) ---")
        print(f"Lexical (RF):      {np.mean(fold_aucs['lexical'])}")
        print(f"Behavioral (RF):   {np.mean(fold_aucs['behavioral'])}")
        print(f"Semantic (XGB):    {np.mean(fold_aucs['semantic'])}")
        print(f"Contextual (LR):   {np.mean(fold_aucs['contextual'])}")

        # Train Meta Learner on OOF
        print("\nTraining Meta-Learner on OOF predictions...")
        self.meta_model.fit(oof_preds, y)
        meta_oof_preds = self.meta_model.predict_proba(oof_preds)[:, 1]
        print(f"Meta-Learner OOF AUC: {roc_auc_score(y, meta_oof_preds)}")

        # Retrain Base Learners on Full Data
        print("\nRetraining Base Learners on full dataset...")

        self.lexical_model.fit(X_dict["lexical"], y)
        self.behavioral_model.fit(X_dict["behavioral"], y)

        # For XGB full retrain, we don't have a validation set for early stopping.
        # We rely on the n_estimators set in config.
        self.semantic_model.fit(X_dict["semantic"], y)

        self.contextual_model.fit(X_dict["contextual"], y)

        self.models_fitted = True
        print("Training complete.")

    def predict_proba(self, X_dict):
        """
        Generates predictions using the stacked ensemble.
        """
        if not self.models_fitted:
            raise RuntimeError("Models not fitted. Call fit() first.")

        n_samples = X_dict["contextual"].shape[0]
        base_preds = np.zeros((n_samples, 4))

        # 1. Lexical
        base_preds[:, 0] = self.lexical_model.predict_proba(X_dict["lexical"])[:, 1]

        # 2. Behavioral
        base_preds[:, 1] = self.behavioral_model.predict_proba(X_dict["behavioral"])[
            :, 1
        ]

        # 3. Semantic
        base_preds[:, 2] = self.semantic_model.predict_proba(X_dict["semantic"])[:, 1]

        # 4. Contextual
        base_preds[:, 3] = self.contextual_model.predict_proba(X_dict["contextual"])[
            :, 1
        ]

        # Meta Prediction
        final_probs = self.meta_model.predict_proba(base_preds)[:, 1]

        return final_probs


def run_stacking_pipeline(train_feats, y_train, test_feats, test_ids):
    """
    Orchestrates the training and prediction process.

    Args:
        train_feats (dict): Dictionary of training features.
        y_train (array-like): Training targets.
        test_feats (dict): Dictionary of test features.
        test_ids (array-like): IDs for the test set.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # Initialize and Train
    stacker = QuadStackingClassifier()
    stacker.fit(train_feats, y_train)

    # Predict on Test
    print("\nGenerating predictions for test set...")
    probs = stacker.predict_proba(test_feats)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: probs})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    return submission_df
