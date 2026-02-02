import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

from library.config import Config
from library.utils import set_seed, save_model, save_submission
from library.data_loader import load_dataset
from library.features import get_features
from library.models import SparseBagger, DenseBooster, StackingMetaLearner


def train_ensemble(load_cached_data=True):
    """
    Orchestrates the training of the Topology-Aware Stacking Ensemble.

    1. Loads data and generates features (Lexical, Behavioral, Semantic).
    2. Performs 5-Fold CV to generate Out-Of-Fold (OOF) predictions for Level 1 models.
    3. Trains a Level 2 Meta-Learner on the OOF predictions.
    4. Retrains Level 1 models on the full training set.
    5. Generates predictions for the test set and saves the submission.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.
    """
    set_seed()

    # ---------------------------------------------------------
    # 1. Data Loading and Feature Generation
    # ---------------------------------------------------------
    print("Loading datasets...")
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    print("Generating features...")
    # get_features handles caching internally based on the boolean flag
    data = get_features(train_df, val_df, test_df, load_cached_data=load_cached_data)

    # Unpack Training Data
    X_train_lexical = data["X_train_lexical"]
    X_train_behavioral = data["X_train_behavioral"]
    X_train_semantic = data["X_train_semantic"]
    y_train = data["y_train"]

    # Unpack Validation Data (Used for Final XGB Early Stopping)
    X_val_semantic = data["X_val_semantic"]
    y_val = data["y_val"]

    # Unpack Test Data
    X_test_lexical = data["X_test_lexical"]
    X_test_behavioral = data["X_test_behavioral"]
    X_test_semantic = data["X_test_semantic"]
    test_ids = data["test_ids"]

    # ---------------------------------------------------------
    # 2. Level 1: Cross-Validation Stacking
    # ---------------------------------------------------------
    print(f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation for Level 1 Models...")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Initialize arrays for Out-Of-Fold predictions
    n_train = y_train.shape[0]
    oof_lexical = np.zeros(n_train)
    oof_behavioral = np.zeros(n_train)
    oof_semantic = np.zeros(n_train)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Prepare Fold Data
        # Lexical View (Sparse)
        X_lex_tr, X_lex_val = X_train_lexical[train_idx], X_train_lexical[val_idx]
        # Behavioral View (Sparse)
        X_beh_tr, X_beh_val = X_train_behavioral[train_idx], X_train_behavioral[val_idx]
        # Semantic View (Dense)
        X_sem_tr, X_sem_val = X_train_semantic[train_idx], X_train_semantic[val_idx]
        # Targets
        y_tr, y_fold_val = y_train[train_idx], y_train[val_idx]

        # A. Train Lexical Bagger (Random Forest)
        print("Training Lexical Bagger (RF)...")
        lex_model = SparseBagger(params=Config.LEXICAL_RF_PARAMS)
        lex_model.fit(X_lex_tr, y_tr)
        pred_lex = lex_model.predict_proba(X_lex_val)
        oof_lexical[val_idx] = pred_lex
        print(f"Fold {fold+1} Lexical AUC: {roc_auc_score(y_fold_val, pred_lex)}")

        # B. Train Behavioral Bagger (Random Forest)
        print("Training Behavioral Bagger (RF)...")
        beh_model = SparseBagger(params=Config.BEHAVIORAL_RF_PARAMS)
        beh_model.fit(X_beh_tr, y_tr)
        pred_beh = beh_model.predict_proba(X_beh_val)
        oof_behavioral[val_idx] = pred_beh
        print(f"Fold {fold+1} Behavioral AUC: {roc_auc_score(y_fold_val, pred_beh)}")

        # C. Train Semantic Booster (XGBoost)
        print("Training Semantic Booster (XGB)...")
        sem_model = DenseBooster(params=Config.SEMANTIC_XGB_PARAMS)
        # Use fold validation set for early stopping
        sem_model.fit(X_sem_tr, y_tr, X_val=X_sem_val, y_val=y_fold_val)
        pred_sem = sem_model.predict_proba(X_sem_val)
        oof_semantic[val_idx] = pred_sem
        print(f"Fold {fold+1} Semantic AUC: {roc_auc_score(y_fold_val, pred_sem)}")

    # Report Overall OOF Performance
    print("\n--- Out-Of-Fold (OOF) Performance ---")
    print(f"Lexical OOF AUC: {roc_auc_score(y_train, oof_lexical)}")
    print(f"Behavioral OOF AUC: {roc_auc_score(y_train, oof_behavioral)}")
    print(f"Semantic OOF AUC: {roc_auc_score(y_train, oof_semantic)}")

    # ---------------------------------------------------------
    # 3. Level 2: Meta-Learner Training
    # ---------------------------------------------------------
    print("\nTraining Level 2 Meta-Learner...")
    # Stack OOF predictions to create meta-features
    X_meta_train = np.column_stack([oof_lexical, oof_behavioral, oof_semantic])

    meta_learner = StackingMetaLearner(params=Config.META_LR_PARAMS)
    meta_learner.fit(X_meta_train, y_train)
    save_model(meta_learner.model, "meta_learner.joblib")

    # ---------------------------------------------------------
    # 4. Final Retraining of Level 1 Models
    # ---------------------------------------------------------
    print("\nRetraining Level 1 Models on Full Training Data...")

    # Retrain Lexical RF
    print("Retraining Lexical RF...")
    final_lex_model = SparseBagger(params=Config.LEXICAL_RF_PARAMS)
    final_lex_model.fit(X_train_lexical, y_train)
    save_model(final_lex_model.model, "lexical_rf.joblib")

    # Retrain Behavioral RF
    print("Retraining Behavioral RF...")
    final_beh_model = SparseBagger(params=Config.BEHAVIORAL_RF_PARAMS)
    final_beh_model.fit(X_train_behavioral, y_train)
    save_model(final_beh_model.model, "behavioral_rf.joblib")

    # Retrain Semantic XGB
    print("Retraining Semantic XGB...")
    final_sem_model = DenseBooster(params=Config.SEMANTIC_XGB_PARAMS)
    # Use the held-out validation set (from metadata) for early stopping
    # This ensures robust convergence without overfitting to the full train set
    final_sem_model.fit(X_train_semantic, y_train, X_val=X_val_semantic, y_val=y_val)
    save_model(final_sem_model.model, "semantic_xgb.joblib")

    # ---------------------------------------------------------
    # 5. Inference and Submission
    # ---------------------------------------------------------
    print("\nGenerating Final Test Predictions...")

    # Generate Level 1 Predictions for Test Set
    test_pred_lex = final_lex_model.predict_proba(X_test_lexical)
    test_pred_beh = final_beh_model.predict_proba(X_test_behavioral)
    test_pred_sem = final_sem_model.predict_proba(X_test_semantic)

    # Stack Level 1 Predictions
    X_meta_test = np.column_stack([test_pred_lex, test_pred_beh, test_pred_sem])

    # Generate Final Level 2 Predictions
    final_test_probs = meta_learner.predict_proba(X_meta_test)

    # Save Submission
    save_submission(test_ids, final_test_probs)
    print("\nTraining Pipeline Completed Successfully.")
