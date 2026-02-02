import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from library.config import ID_COL, TARGET_COL, SUBMISSION_PATH, SEED, N_FOLDS
from library.utils import set_seed, timer
from library.data_loader import load_dataset, get_stratified_cv
from library.feature_engineering import FeaturePipeline, generate_data_views
from library.model_definitions import LexicalRF, BehavioralRF, SemanticXGB, MetaLearner


def train_stacking_ensemble(load_cached_data=True):
    """
    Orchestrates the training of the High-Capacity Topology-Matched Stacking Ensemble.

    Steps:
    1. Load and combine Train/Val datasets.
    2. Generate feature views (Lexical, Behavioral, Semantic).
    3. Perform 5-Fold CV to generate OOF predictions for Level 1 models.
    4. Train Level 2 Meta-Learner on OOF predictions.
    5. Retrain Level 1 models on full training data.
    6. Generate Test predictions and save submission.
    """
    set_seed(SEED)

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    with timer("Loading Data"):
        # Load base datasets (leakage columns already removed by loader)
        train_df_part, val_df_part, test_df = load_dataset(
            load_cached_data=load_cached_data
        )

        # Combine provided train and val splits into a single full training set
        # for Cross-Validation and final retraining.
        full_train_df = pd.concat([train_df_part, val_df_part], axis=0).reset_index(
            drop=True
        )

        print(f"Full Training Set Shape: {full_train_df.shape}")
        print(f"Test Set Shape: {test_df.shape}")

        # Extract targets and IDs
        y_full = full_train_df[TARGET_COL].values
        test_ids = test_df[ID_COL].values

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    pipeline = FeaturePipeline()

    # Generate features for Full Train (Fit pipeline here)
    # We use a custom stage name 'train_full' to cache these specific features
    train_views = generate_data_views(
        full_train_df,
        stage="train_full",
        pipeline=pipeline,
        fit=True,
        load_cached_data=load_cached_data,
    )

    # Generate features for Test (Transform only)
    test_views = generate_data_views(
        test_df,
        stage="test",
        pipeline=pipeline,
        fit=False,
        load_cached_data=load_cached_data,
    )

    # ---------------------------------------------------------
    # 3. Level 1: Cross-Validation & OOF Generation
    # ---------------------------------------------------------
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation for Level 1 Models...")

    skf = get_stratified_cv(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Initialize OOF prediction matrix: (N_samples, N_models)
    # Models: 0=LexicalRF, 1=BehavioralRF, 2=SemanticXGB
    oof_preds = np.zeros((len(full_train_df), 3))

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, y_full)):
        print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")

        # Slice Targets
        y_train_fold = y_full[train_idx]
        y_val_fold = y_full[val_idx]

        # Slice Features (Sparse matrices need specific slicing)
        X_lex_train = train_views["lexical"][train_idx]
        X_lex_val = train_views["lexical"][val_idx]

        X_beh_train = train_views["behavioral"][train_idx]
        X_beh_val = train_views["behavioral"][val_idx]

        X_sem_train = train_views["semantic"][train_idx]
        X_sem_val = train_views["semantic"][val_idx]

        # --- Model 1: Lexical RF ---
        lex_rf = LexicalRF()
        lex_rf.fit(X_lex_train, y_train_fold)
        p_lex = lex_rf.predict_proba(X_lex_val)[:, 1]
        score_lex = roc_auc_score(y_val_fold, p_lex)
        print(f"LexicalRF AUC: {score_lex}")

        # --- Model 2: Behavioral RF ---
        beh_rf = BehavioralRF()
        beh_rf.fit(X_beh_train, y_train_fold)
        p_beh = beh_rf.predict_proba(X_beh_val)[:, 1]
        score_beh = roc_auc_score(y_val_fold, p_beh)
        print(f"BehavioralRF AUC: {score_beh}")

        # --- Model 3: Semantic XGB ---
        sem_xgb = SemanticXGB()
        # Pass validation set for early stopping
        sem_xgb.fit(X_sem_train, y_train_fold, X_val=X_sem_val, y_val=y_val_fold)
        p_sem = sem_xgb.predict_proba(X_sem_val)[:, 1]
        score_sem = roc_auc_score(y_val_fold, p_sem)
        print(f"SemanticXGB AUC: {score_sem}")

        # Store OOF predictions
        oof_preds[val_idx, 0] = p_lex
        oof_preds[val_idx, 1] = p_beh
        oof_preds[val_idx, 2] = p_sem

        # Calculate simple average ensemble score for this fold
        p_avg = (p_lex + p_beh + p_sem) / 3
        score_avg = roc_auc_score(y_val_fold, p_avg)
        fold_scores.append(score_avg)
        print(f"Fold {fold+1} Average Ensemble AUC: {score_avg}")

    print(f"\nMean CV AUC (Simple Ensemble): {np.mean(fold_scores)}")

    # ---------------------------------------------------------
    # 4. Level 2: Train Meta-Learner
    # ---------------------------------------------------------
    print("\nTraining Level 2 Meta-Learner on OOF predictions...")
    meta_learner = MetaLearner()
    meta_learner.fit(oof_preds, y_full)

    # Log coefficients to understand model contribution
    if hasattr(meta_learner.model, "coef_"):
        coefs = meta_learner.model.coef_[0]
        print(
            f"Meta-Learner Coefficients: Lexical={coefs[0]}, Behavioral={coefs[1]}, Semantic={coefs[2]}"
        )

    # ---------------------------------------------------------
    # 5. Retrain Level 1 Models on Full Data
    # ---------------------------------------------------------
    print("\nRetraining Level 1 Models on Full Training Data...")

    with timer("Retraining LexicalRF"):
        final_lex_rf = LexicalRF()
        final_lex_rf.fit(train_views["lexical"], y_full)

    with timer("Retraining BehavioralRF"):
        final_beh_rf = BehavioralRF()
        final_beh_rf.fit(train_views["behavioral"], y_full)

    with timer("Retraining SemanticXGB"):
        final_sem_xgb = SemanticXGB()
        # No early stopping possible on full train as we have no holdout
        final_sem_xgb.fit(train_views["semantic"], y_full)

    # ---------------------------------------------------------
    # 6. Generate Test Predictions
    # ---------------------------------------------------------
    print("\nGenerating Test Predictions...")

    # Get Level 1 predictions for Test set
    test_p_lex = final_lex_rf.predict_proba(test_views["lexical"])[:, 1]
    test_p_beh = final_beh_rf.predict_proba(test_views["behavioral"])[:, 1]
    test_p_sem = final_sem_xgb.predict_proba(test_views["semantic"])[:, 1]

    # Stack predictions for Level 2
    X_test_level2 = np.column_stack([test_p_lex, test_p_beh, test_p_sem])

    # Get Final predictions from Meta-Learner
    final_probs = meta_learner.predict_proba(X_test_level2)[:, 1]

    # ---------------------------------------------------------
    # 7. Save Submission
    # ---------------------------------------------------------
    print(f"Saving submission to {SUBMISSION_PATH}...")

    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Training and Prediction Pipeline Complete.")
