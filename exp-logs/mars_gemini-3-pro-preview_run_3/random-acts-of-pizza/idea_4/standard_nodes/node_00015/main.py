import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import set_seed, save_predictions, load_data, get_cached_data
from library.feature_engineering import FeatureEngineer
from library.dataset import _compute_transformer_data, PizzaDataset
from library.models import (
    get_lexical_model,
    get_style_model,
    get_meta_model,
    SemanticFineTuner,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting execution of runfile.py...")

    # 2. Data Loading & Feature Engineering
    print("\n--- Phase 1: Data Preparation ---")
    fe = FeatureEngineer()

    # Process tabular/text features
    data = fe.process_data(load_cached_data=True, debug=Config.DEBUG)

    # Process transformer data (tokenization)
    # We need to manually call the helper to get the raw arrays
    train_df = load_data(Config.TRAIN_PATH, debug=Config.DEBUG)
    val_df = load_data(Config.VAL_PATH, debug=Config.DEBUG)
    test_df = load_data(Config.TEST_PATH, debug=Config.DEBUG)

    suffix = "_debug" if Config.DEBUG else ""
    trans_data = get_cached_data(
        _compute_transformer_data,
        f"transformer_data{suffix}",
        load_cached_data=True,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # Prepare Feature Matrices
    # Lexical: TF-IDF + Meta
    X_lex_train = np.hstack([data["train"]["lexical"], data["train"]["meta"]])
    X_lex_val = np.hstack([data["val"]["lexical"], data["val"]["meta"]])
    X_lex_test = np.hstack([data["test"]["lexical"], data["test"]["meta"]])

    # Style: Style + Meta
    X_style_train = np.hstack([data["train"]["style"], data["train"]["meta"]])
    X_style_val = np.hstack([data["val"]["style"], data["val"]["meta"]])
    X_style_test = np.hstack([data["test"]["style"], data["test"]["meta"]])

    # Targets
    y_train = data["train"]["y"]
    y_val = data["val"]["y"]

    # Transformer Inputs
    train_input_ids = trans_data["train_input_ids"]
    train_masks = trans_data["train_attention_mask"]
    val_input_ids = trans_data["val_input_ids"]
    val_masks = trans_data["val_attention_mask"]
    test_input_ids = trans_data["test_input_ids"]
    test_masks = trans_data["test_attention_mask"]

    # 3. Level 1 Training (CV on Train for Stacking)
    print("\n--- Phase 2: Level 1 Cross-Validation (Generating OOF) ---")
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # OOF Arrays
    oof_lexical = np.zeros(len(y_train))
    oof_style = np.zeros(len(y_train))
    oof_semantic = np.zeros(len(y_train))

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_lex_train, y_train)):
        print(f"Processing Fold {fold + 1}/{n_folds}...")

        # --- Lexical ---
        rf = get_lexical_model()
        rf.fit(X_lex_train[t_idx], y_train[t_idx])
        oof_lexical[v_idx] = rf.predict_proba(X_lex_train[v_idx])[:, 1]

        # --- Style ---
        xgb = get_style_model()
        xgb.fit(X_style_train[t_idx], y_train[t_idx])
        oof_style[v_idx] = xgb.predict_proba(X_style_train[v_idx])[:, 1]

        # --- Semantic ---
        # Create datasets for this fold
        fold_train_ds = PizzaDataset(
            train_input_ids[t_idx], train_masks[t_idx], y_train[t_idx]
        )
        fold_val_ds = PizzaDataset(
            train_input_ids[v_idx], train_masks[v_idx], y_train[v_idx]
        )

        fold_train_loader = DataLoader(
            fold_train_ds,
            batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        fold_val_loader = DataLoader(
            fold_val_ds,
            batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        bert = SemanticFineTuner()
        # Reduce epochs for CV to speed up baseline
        bert.epochs = 2
        bert.fit(fold_train_loader, fold_val_loader)
        oof_semantic[v_idx] = bert.predict_proba(fold_val_loader)

    # 4. Meta-Learner Training
    print("\n--- Phase 3: Meta-Learner Training ---")
    X_oof = np.column_stack([oof_lexical, oof_style, oof_semantic])
    meta_model = get_meta_model()
    meta_model.fit(X_oof, y_train)

    oof_auc = roc_auc_score(y_train, meta_model.predict_proba(X_oof)[:, 1])
    print(f"Meta-Learner OOF AUC on Train: {oof_auc:.6f}")

    # 5. Full Training & Validation Prediction
    print("\n--- Phase 4: Full Training & Validation ---")

    # --- Lexical ---
    print("Retraining Lexical Model...")
    rf_full = get_lexical_model()
    rf_full.fit(X_lex_train, y_train)
    val_pred_lex = rf_full.predict_proba(X_lex_val)[:, 1]
    test_pred_lex = rf_full.predict_proba(X_lex_test)[:, 1]

    # --- Style ---
    print("Retraining Style Model...")
    xgb_full = get_style_model()
    xgb_full.fit(X_style_train, y_train)
    val_pred_style = xgb_full.predict_proba(X_style_val)[:, 1]
    test_pred_style = xgb_full.predict_proba(X_style_test)[:, 1]

    # --- Semantic ---
    print("Retraining Semantic Model...")
    # Use full Train for training, Val for Early Stopping
    train_ds_full = PizzaDataset(train_input_ids, train_masks, y_train)
    val_ds_full = PizzaDataset(val_input_ids, val_masks, y_val)
    test_ds_full = PizzaDataset(test_input_ids, test_masks, labels=None)

    train_loader_full = DataLoader(
        train_ds_full,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader_full = DataLoader(
        test_ds_full,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    bert_full = SemanticFineTuner()
    bert_full.epochs = 3  # Use full epochs for final model
    bert_full.fit(train_loader_full, val_loader_full)

    val_pred_sem = bert_full.predict_proba(val_loader_full)
    test_pred_sem = bert_full.predict_proba(test_loader_full)

    # 6. Evaluation
    X_val_meta = np.column_stack([val_pred_lex, val_pred_style, val_pred_sem])
    val_final_preds = meta_model.predict_proba(X_val_meta)[:, 1]

    final_auc = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n--- Phase 5: Failure Analysis ---")
    # Calculate residuals
    residuals = np.abs(y_val - val_final_preds)

    # Correlate residuals with metadata features
    # We use the 'meta' features from the validation set
    meta_features = data["val"]["meta"]
    # We need column names for better reporting, but we have a numpy array.
    # We can infer indices or just print top correlations by index.
    # Let's map back to the safe cols list from FeatureEngineer
    safe_cols = [
        "account_age",
        "days_since_first_post",
        "comments_at_req",
        "comments_in_raop",
        "posts_at_req",
        "posts_on_raop",
        "subreddits",
        "up_minus_down",
        "up_plus_down",
    ]

    correlations = []
    for i in range(meta_features.shape[1]):
        if i < len(safe_cols):
            feat_name = safe_cols[i]
        else:
            feat_name = f"Feature_{i}"

        corr = np.corrcoef(residuals, meta_features[:, i])[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Correlations with Error (Residuals):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    threshold = 0.6913548345419015
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )
        X_test_meta = np.column_stack([test_pred_lex, test_pred_style, test_pred_sem])
        final_test_preds = meta_model.predict_proba(X_test_meta)[:, 1]

        save_predictions(data["test"]["ids"], final_test_preds)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric ({final_auc:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
