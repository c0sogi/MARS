import numpy as np
import pandas as pd
from scipy.stats import entropy
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from library.utils import compute_log_loss
from library.config import Config


def extract_meta_features(probs_linear, probs_trans, texts):
    """
    Extracts meta-features for stacking.
    Includes raw probabilities, uncertainty metrics, and text statistics.
    Cite solution_lesson_node_00008 (Meta-Feature Stacking)
    Cite solution_lesson_node_00009 (Uncertainty Signals)
    """
    # 1. Uncertainty Features from Linear Model
    lin_entropy = entropy(probs_linear, axis=1)
    lin_max = np.max(probs_linear, axis=1)
    lin_std = np.std(probs_linear, axis=1)

    # 2. Uncertainty Features from Transformer Model
    trans_entropy = entropy(probs_trans, axis=1)
    trans_max = np.max(probs_trans, axis=1)
    trans_std = np.std(probs_trans, axis=1)

    # 3. Text Features
    # Ensure texts are strings
    texts = texts.fillna("").astype(str).values
    char_len = np.array([len(t) for t in texts])
    word_count = np.array([len(t.split()) for t in texts])
    avg_word_len = np.array(
        [
            np.mean([len(w) for w in t.split()]) if len(t.split()) > 0 else 0
            for t in texts
        ]
    )

    # Punctuation density
    puncts = set("!?,.;:-'\"")
    punct_density = np.array(
        [sum(1 for c in t if c in puncts) / (len(t) + 1e-6) for t in texts]
    )

    # Combine all features
    features = np.column_stack(
        [
            probs_linear,  # 3 cols
            probs_trans,  # 3 cols
            lin_entropy,
            lin_max,
            lin_std,
            trans_entropy,
            trans_max,
            trans_std,
            char_len,
            word_count,
            avg_word_len,
            punct_density,
        ]
    )

    return features


def run_ensemble(
    val_probs_linear,
    test_probs_linear,
    val_probs_transformer,
    test_probs_transformer,
    y_val,
    val_texts,
    test_texts,
):
    """
    Orchestrates the ensemble process using Hold-Out Blending with XGBoost.
    Cite solution_lesson_node_00010 (Hold-Out Blending)
    """
    print("--- Starting Ensemble Pipeline (Stacking) ---")

    # 1. Generate Meta-Features
    print("Extracting meta-features for Validation set...")
    X_meta_val = extract_meta_features(
        val_probs_linear, val_probs_transformer, val_texts
    )

    print("Extracting meta-features for Test set...")
    X_meta_test = extract_meta_features(
        test_probs_linear, test_probs_transformer, test_texts
    )

    # 2. Generate Out-of-Fold Predictions for Validation Set
    print("Generating OOF predictions via Stratified K-Fold...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    # Array to store OOF predictions
    val_preds_oof = np.zeros((X_meta_val.shape[0], 3))

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_val, y_val)):
        X_train_fold, y_train_fold = X_meta_val[train_idx], y_val[train_idx]
        X_val_fold = X_meta_val[val_idx]

        # Initialize and train meta-learner for this fold
        clf_fold = XGBClassifier(**Config.XGB_PARAMS)
        clf_fold.fit(X_train_fold, y_train_fold, verbose=False)

        # Predict on the hold-out fold
        val_preds_oof[val_idx] = clf_fold.predict_proba(X_val_fold)

    # 3. Train Final Meta-Learner on Full Validation Set
    print("Training final XGBoost Meta-Learner on full validation set...")
    clf_final = XGBClassifier(**Config.XGB_PARAMS)
    clf_final.fit(X_meta_val, y_val, verbose=False)

    # 4. Generate Predictions for Test Set
    print("Generating final test predictions...")
    test_preds = clf_final.predict_proba(X_meta_test)

    # 5. Verify Validation Score (using OOF predictions)
    val_loss = compute_log_loss(y_val, val_preds_oof, labels=[0, 1, 2])
    print(f"Final Ensemble Validation Log Loss (OOF): {val_loss}")

    print("--- Ensemble Pipeline Complete ---")
    return val_preds_oof, test_preds
