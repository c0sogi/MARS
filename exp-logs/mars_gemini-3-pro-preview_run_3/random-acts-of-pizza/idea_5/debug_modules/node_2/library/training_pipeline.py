import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import N_FOLDS, SEED
from library.utils import set_seed, timer, print_metrics
from library.model_definitions import (
    LexicalBagger,
    SemanticBagger,
    CommunityBooster,
    StackingMetaLearner,
)


def train_ensemble(X_train_dict, y_train):
    """
    Performs 5-Fold Cross-Validation Stacking to train the ensemble.

    The process involves:
    1. Splitting the training data into stratified folds.
    2. Training Level 1 Base Learners (Lexical, Semantic, Community) on (K-1) folds
       and generating Out-Of-Fold (OOF) predictions on the hold-out fold.
    3. Training a Level 2 Meta-Learner (Logistic Regression) on the OOF predictions.
    4. Retraining all Level 1 Base Learners on the full training dataset.

    Args:
        X_train_dict (dict): Dictionary containing feature matrices for different views.
                             Keys: 'lexical', 'semantic', 'community'.
                             Values: Sparse or Dense matrices/arrays.
        y_train (pd.Series): Target labels corresponding to the training data.

    Returns:
        dict: A dictionary containing the trained model artifacts:
              {
                  'lexical': <trained LexicalBagger>,
                  'semantic': <trained SemanticBagger>,
                  'community': <trained CommunityBooster>,
                  'meta': <trained StackingMetaLearner>
              }
    """
    set_seed(SEED)

    n_samples = len(y_train)
    # Initialize OOF prediction matrix: [n_samples, 3 models]
    # Columns: 0=Lexical, 1=Semantic, 2=Community
    oof_preds = np.zeros((n_samples, 3))

    # Stratified K-Fold for stable validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"Starting {N_FOLDS}-Fold Cross-Validation Stacking...")

    # --- Phase 1: Cross-Validation & OOF Generation ---
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(n_samples), y_train)
    ):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Slice features for current fold
        # Handles both sparse matrices (scipy) and dense arrays (numpy)
        X_tr_fold = {k: v[train_idx] for k, v in X_train_dict.items()}
        X_val_fold = {k: v[val_idx] for k, v in X_train_dict.items()}

        y_tr_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        # 1. Lexical Bagger (Random Forest)
        with timer("Lexical Bagger (Fold)"):
            lex_model = LexicalBagger()
            lex_model.fit(X_tr_fold, y_tr_fold)
            p_lex = lex_model.predict_proba(X_val_fold)
            oof_preds[val_idx, 0] = p_lex

        # 2. Semantic Bagger (Random Forest)
        with timer("Semantic Bagger (Fold)"):
            sem_model = SemanticBagger()
            sem_model.fit(X_tr_fold, y_tr_fold)
            p_sem = sem_model.predict_proba(X_val_fold)
            oof_preds[val_idx, 1] = p_sem

        # 3. Community Booster (XGBoost)
        with timer("Community Booster (Fold)"):
            com_model = CommunityBooster()
            # XGBoost allows early stopping using the validation fold
            com_model.fit(
                X_tr_fold,
                y_tr_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=50,
                verbose=False,
            )
            p_com = com_model.predict_proba(X_val_fold)
            oof_preds[val_idx, 2] = p_com

        # Print Fold Metrics
        print(f"  Fold {fold+1} AUC - Lexical: {roc_auc_score(y_val_fold, p_lex)}")
        print(f"  Fold {fold+1} AUC - Semantic: {roc_auc_score(y_val_fold, p_sem)}")
        print(f"  Fold {fold+1} AUC - Community: {roc_auc_score(y_val_fold, p_com)}")

    # --- Phase 2: OOF Evaluation ---
    print("\n--- Out-Of-Fold (OOF) Performance ---")
    auc_lex_oof = roc_auc_score(y_train, oof_preds[:, 0])
    auc_sem_oof = roc_auc_score(y_train, oof_preds[:, 1])
    auc_com_oof = roc_auc_score(y_train, oof_preds[:, 2])

    print_metrics(
        {
            "OOF Lexical AUC": auc_lex_oof,
            "OOF Semantic AUC": auc_sem_oof,
            "OOF Community AUC": auc_com_oof,
        }
    )

    # --- Phase 3: Meta-Learner Training ---
    print("\nTraining Level 2 Meta-Learner on OOF predictions...")
    with timer("Meta-Learner Training"):
        meta_learner = StackingMetaLearner()
        meta_learner.fit(oof_preds, y_train)

    if hasattr(meta_learner.model, "coef_"):
        print(f"Meta-Learner Coefficients: {meta_learner.model.coef_}")
        print(f"Meta-Learner Intercept: {meta_learner.model.intercept_}")

    # --- Phase 4: Final Retraining on Full Data ---
    print("\nRetraining Level 1 models on full training dataset...")

    with timer("Lexical Bagger (Full Retrain)"):
        final_lex = LexicalBagger()
        final_lex.fit(X_train_dict, y_train)

    with timer("Semantic Bagger (Full Retrain)"):
        final_sem = SemanticBagger()
        final_sem.fit(X_train_dict, y_train)

    with timer("Community Booster (Full Retrain)"):
        final_com = CommunityBooster()
        # No early stopping for full retrain (no validation set), runs for n_estimators
        final_com.fit(X_train_dict, y_train, verbose=False)

    print("Ensemble training complete.")

    return {
        "lexical": final_lex,
        "semantic": final_sem,
        "community": final_com,
        "meta": meta_learner,
    }
