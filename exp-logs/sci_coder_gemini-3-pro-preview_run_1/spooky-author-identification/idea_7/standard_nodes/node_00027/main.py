import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.dataset import create_folds, AuthorDataset
from library.features import extract_meta_features
from library.model_transformer import run_expert_a, DebertaClassifier
from library.model_linear import run_expert_b
from library.meta_learner import run_meta_learner

# ------------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# ------------------------------------------------------------------------------
# Reduce epochs to ensure execution finishes within time limit
Config.EPOCHS = 2
# Increase batch size slightly for A100 efficiency
Config.TRAIN_BATCH_SIZE = 8
Config.GRADIENT_ACCUMULATION_STEPS = 4


def get_expert_a_val_preds(val_df):
    """
    Generates predictions for the validation set using the 5 saved DeBERTa checkpoints.
    """
    print("\n[Validation] Generating Expert A predictions...")
    device = Config.DEVICE
    val_texts = val_df["text"].values
    val_dataset = AuthorDataset(texts=val_texts, labels=None)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    fold_preds = []

    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"expert_a_fold_{fold}.pt"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping fold.")
            continue

        model = DebertaClassifier(Config.MODEL_NAME).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(input_ids, attention_mask)
                    logits = outputs["logits"]

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds.append(probs)

        fold_preds.append(np.concatenate(preds, axis=0))

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not fold_preds:
        raise RuntimeError("No Expert A checkpoints found!")

    # Average predictions (Bagging)
    avg_preds = np.mean(fold_preds, axis=0)
    return avg_preds


def get_expert_b_val_preds(df_folds, val_df):
    """
    Re-runs Expert B training (TF-IDF + LR) to generate predictions for the validation set.
    """
    print("\n[Validation] Generating Expert B predictions...")
    val_texts = val_df["text"].values.astype(str)
    fold_preds = []

    # Replicate the 5-fold training loop
    for fold in range(Config.N_FOLDS):
        # Split Data
        train_mask = df_folds["fold"] != fold
        X_train_text = df_folds.loc[train_mask, "text"].values.astype(str)
        y_train = df_folds.loc[train_mask, "author"].map(Config.LABEL2ID).values

        # TF-IDF Vectorizers
        word_vectorizer = TfidfVectorizer(
            ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
            min_df=Config.TFIDF_MIN_DF,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
            min_df=Config.TFIDF_MIN_DF,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            analyzer="char",
        )

        # Fit on Train
        X_train_word = word_vectorizer.fit_transform(X_train_text)
        X_train_char = char_vectorizer.fit_transform(X_train_text)
        X_train_feats = hstack([X_train_word, X_train_char])

        # Transform Validation
        X_val_word = word_vectorizer.transform(val_texts)
        X_val_char = char_vectorizer.transform(val_texts)
        X_val_feats = hstack([X_val_word, X_val_char])

        # Train LR
        clf = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            C=1.0,
            random_state=Config.SEED,
            n_jobs=-1,
            max_iter=1000,
        )
        clf.fit(X_train_feats, y_train)

        # Predict
        probs = clf.predict_proba(X_val_feats)
        fold_preds.append(probs)

        del clf, word_vectorizer, char_vectorizer, X_train_feats, X_val_feats
        gc.collect()

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)
    return avg_preds


def perform_failure_analysis(val_df, y_true_indices, y_pred_probs):
    """
    Analyzes correlation between error magnitude and meta-features.
    """
    print("\n[Failure Analysis] Computing correlations...")

    # Calculate error: -log(p_true)
    # Select probability of the true class
    n_samples = len(y_true_indices)
    true_probs = y_pred_probs[np.arange(n_samples), y_true_indices]
    # Clip for stability
    true_probs = np.clip(true_probs, 1e-15, 1.0)
    errors = -np.log(true_probs)

    # Extract features
    meta_df = extract_meta_features(val_df)

    # Compute correlations
    features = ["char_len", "word_count", "punct_density"]
    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        corr = np.corrcoef(errors, meta_df[feat])[0, 1]
        print(f"  {feat}: {corr:.4f}")


def main():
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 1. Run Experts (Train on Train Set, Predict OOF & Test)
    # --------------------------------------------------------------------------
    print("=== Starting Expert A (DeBERTa) ===")
    oof_a, test_a = run_expert_a(load_cached_data=True, debug=Config.DEBUG)

    print("\n=== Starting Expert B (Linear) ===")
    oof_b, test_b = run_expert_b(load_cached_data=True, debug=Config.DEBUG)

    # --------------------------------------------------------------------------
    # 2. Validation Assessment (Hold-out Set)
    # --------------------------------------------------------------------------
    print("\n=== Validation Assessment ===")
    # Load Validation Data
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Validation data not found at {Config.VAL_DATA_PATH}")

    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    y_val_true = val_df["author"].map(Config.LABEL2ID).values

    # Load Training Folds (needed for Expert B retraining and Meta training)
    df_folds = create_folds(load_cached_data=True, debug=Config.DEBUG)
    y_train_true = df_folds["author"].map(Config.LABEL2ID).values

    # 2a. Expert A Validation Predictions
    val_pred_a = get_expert_a_val_preds(val_df)

    # 2b. Expert B Validation Predictions
    val_pred_b = get_expert_b_val_preds(df_folds, val_df)

    # 2c. Meta Features
    print("[Validation] Extracting meta-features...")
    meta_train = extract_meta_features(df_folds).values.astype(np.float32)
    meta_val = extract_meta_features(val_df).values.astype(np.float32)

    # 2d. Train Meta-Learner on OOFs and Predict on Validation
    print("[Validation] Training Meta-Learner for evaluation...")
    X_train_stack = np.hstack([oof_a, oof_b, meta_train])
    X_val_stack = np.hstack([val_pred_a, val_pred_b, meta_val])

    # Train XGBoost on full training OOFs
    xgb_params = Config.XGB_PARAMS.copy()
    xgb_params["n_estimators"] = 500  # Sufficient for convergence on this size

    meta_model = xgb.XGBClassifier(**xgb_params)
    meta_model.fit(X_train_stack, y_train_true, verbose=False)

    # Predict on Validation
    val_final_probs = meta_model.predict_proba(X_val_stack)

    # Compute Metric
    final_metric = compute_log_loss(y_val_true, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 3. Failure Analysis
    # --------------------------------------------------------------------------
    perform_failure_analysis(val_df, y_val_true, val_final_probs)

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.25336663725445785

    if final_metric < THRESHOLD:
        print("\n=== Generating Submission ===")
        # run_meta_learner handles the final training on OOFs and prediction on Test
        run_meta_learner(oof_a, test_a, oof_b, test_b, debug=Config.DEBUG)
    else:
        print(
            f"\nValidation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
