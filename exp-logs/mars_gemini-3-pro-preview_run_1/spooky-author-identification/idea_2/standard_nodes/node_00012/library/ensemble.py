import numpy as np
import pandas as pd
from scipy.stats import entropy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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

    # 2. Train Meta-Learner (XGBoost)
    print("Training XGBoost Meta-Learner...")
    clf = XGBClassifier(**Config.XGB_PARAMS)

    # Use part of validation for early stopping to prevent overfitting the meta-learner
    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
        X_meta_val, y_val, test_size=0.2, random_state=Config.SEED, stratify=y_val
    )

    clf.fit(
        X_train_meta, y_train_meta, eval_set=[(X_val_meta, y_val_meta)], verbose=False
    )

    # 3. Generate Predictions
    print("Generating final predictions...")
    val_preds = clf.predict_proba(X_meta_val)
    test_preds = clf.predict_proba(X_meta_test)

    # 4. Verify Validation Score
    val_loss = compute_log_loss(y_val, val_preds, labels=[0, 1, 2])
    print(f"Final Ensemble Validation Log Loss: {val_loss}")

    print("--- Ensemble Pipeline Complete ---")
    return val_preds, test_preds
