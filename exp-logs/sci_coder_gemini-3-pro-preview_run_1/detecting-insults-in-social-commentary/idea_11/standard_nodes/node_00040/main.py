import pandas as pd
import numpy as np
import torch
import os
import gc
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import InsultDataset
from library.trainer import train_fold, predict
from library.model import HybridModel
from library.features import get_fold_features


def main():
    # 1. Setup & Initialization
    seed_everything(Config.seed)

    print("Loading Metadata...")
    # Load Training Metadata (for CV)
    df_train_meta = pd.read_csv(Config.train_path)

    # Load Holdout Data (Validation Set - strictly for final evaluation)
    df_holdout = pd.read_csv(Config.val_path)
    holdout_texts = df_holdout["Comment"].fillna("").tolist()
    holdout_labels = df_holdout["Insult"].values

    # Load Test Data (for Submission)
    df_test = pd.read_csv(Config.test_path)
    test_texts = df_test["Comment"].fillna("").tolist()

    # Combine Holdout and Test for SVD processing
    # This ensures we generate features for both sets using the transformation
    # fitted on the training fold, without fitting on holdout/test (no leakage).
    combined_test_texts = holdout_texts + test_texts
    split_point = len(holdout_texts)

    # Initialize Storage Arrays
    # OOF Predictions: aligned with df_train_meta
    oof_preds_a = np.zeros(len(df_train_meta))
    oof_preds_b = np.zeros(len(df_train_meta))

    # Holdout Predictions: Accumulate sum to average later
    holdout_preds_a = np.zeros(len(df_holdout))
    holdout_preds_b = np.zeros(len(df_holdout))

    # Test Predictions: Accumulate sum to average later
    test_preds_a = np.zeros(len(df_test))
    test_preds_b = np.zeros(len(df_test))

    # Initialize Tokenizers
    tokenizer_a = AutoTokenizer.from_pretrained(Config.model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(Config.model_b_name)

    # 2. Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    # We use the index and label from train metadata to define folds
    splits = list(skf.split(df_train_meta.index, df_train_meta["Insult"]))

    for fold in range(Config.n_folds):
        print(f"\n{'='*20} Processing Fold {fold}/{Config.n_folds - 1} {'='*20}")

        train_idx, val_idx = splits[fold]

        # Extract Texts for this Fold
        fold_train_texts = df_train_meta.iloc[train_idx]["Comment"].fillna("").tolist()
        fold_val_texts = df_train_meta.iloc[val_idx]["Comment"].fillna("").tolist()
        fold_train_labels = df_train_meta.iloc[train_idx]["Insult"].values
        fold_val_labels = df_train_meta.iloc[val_idx]["Insult"].values

        # Generate SVD Features
        # We pass combined_test_texts to get transformed features for both holdout and test
        # Note: load_cached_data=False ensures we re-compute for the combined set
        # instead of loading potentially stale cache from a run with just test set.
        train_svd, val_svd, combined_test_svd = get_fold_features(
            fold,
            fold_train_texts,
            fold_val_texts,
            combined_test_texts,
            load_cached_data=False,
        )

        # Split the combined SVD features back into Holdout and Test
        holdout_svd = combined_test_svd[:split_point]
        test_svd = combined_test_svd[split_point:]

        # --- Model A: DeBERTa-v3-Large ---
        print(f"--- Training Model A: {Config.model_a_name} ---")

        # Create Datasets
        ds_train_a = InsultDataset(
            fold_train_texts, train_svd, tokenizer_a, fold_train_labels
        )
        ds_val_a = InsultDataset(fold_val_texts, val_svd, tokenizer_a, fold_val_labels)
        ds_holdout_a = InsultDataset(
            holdout_texts, holdout_svd, tokenizer_a, holdout_labels
        )
        ds_test_a = InsultDataset(test_texts, test_svd, tokenizer_a, None)

        # Train and get OOF for this fold
        fold_oof_a = train_fold(fold, Config.model_a_name, ds_train_a, ds_val_a)
        oof_preds_a[val_idx] = fold_oof_a

        # Inference on Holdout & Test using the trained model
        # Load the best checkpoint saved by train_fold
        model_a = HybridModel(Config.model_a_name, pretrained=False)
        model_path_a = os.path.join(
            Config.model_dir, f"{Config.model_a_name.replace('/', '_')}_fold_{fold}.bin"
        )
        model_a.load_state_dict(torch.load(model_path_a, map_location=Config.device))
        model_a.to(Config.device)

        holdout_loader_a = DataLoader(
            ds_holdout_a,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )
        test_loader_a = DataLoader(
            ds_test_a,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        holdout_preds_a += (
            predict(model_a, holdout_loader_a, Config.device) / Config.n_folds
        )
        test_preds_a += predict(model_a, test_loader_a, Config.device) / Config.n_folds

        # Cleanup Model A
        del model_a, ds_train_a, ds_val_a, ds_holdout_a, ds_test_a
        gc.collect()
        torch.cuda.empty_cache()

        # --- Model B: RoBERTa-Large ---
        print(f"--- Training Model B: {Config.model_b_name} ---")

        # Create Datasets
        ds_train_b = InsultDataset(
            fold_train_texts, train_svd, tokenizer_b, fold_train_labels
        )
        ds_val_b = InsultDataset(fold_val_texts, val_svd, tokenizer_b, fold_val_labels)
        ds_holdout_b = InsultDataset(
            holdout_texts, holdout_svd, tokenizer_b, holdout_labels
        )
        ds_test_b = InsultDataset(test_texts, test_svd, tokenizer_b, None)

        # Train and get OOF for this fold
        fold_oof_b = train_fold(fold, Config.model_b_name, ds_train_b, ds_val_b)
        oof_preds_b[val_idx] = fold_oof_b

        # Inference on Holdout & Test
        model_b = HybridModel(Config.model_b_name, pretrained=False)
        model_path_b = os.path.join(
            Config.model_dir, f"{Config.model_b_name.replace('/', '_')}_fold_{fold}.bin"
        )
        model_b.load_state_dict(torch.load(model_path_b, map_location=Config.device))
        model_b.to(Config.device)

        holdout_loader_b = DataLoader(
            ds_holdout_b,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )
        test_loader_b = DataLoader(
            ds_test_b,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        holdout_preds_b += (
            predict(model_b, holdout_loader_b, Config.device) / Config.n_folds
        )
        test_preds_b += predict(model_b, test_loader_b, Config.device) / Config.n_folds

        # Cleanup Model B
        del model_b, ds_train_b, ds_val_b, ds_holdout_b, ds_test_b
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Meta-Learner Stacking
    print("\n=== Training Meta-Learner (Ridge Stacking) ===")

    # Prepare Meta-Features
    # Train Meta: OOF predictions from A and B
    X_train_meta = np.vstack([oof_preds_a, oof_preds_b]).T
    y_train_meta = df_train_meta["Insult"].values

    # Holdout Meta: Averaged predictions from A and B
    X_holdout_meta = np.vstack([holdout_preds_a, holdout_preds_b]).T

    # Test Meta: Averaged predictions from A and B
    X_test_meta = np.vstack([test_preds_a, test_preds_b]).T

    # Train Ridge Regression
    meta_model = Ridge(alpha=Config.meta_alpha, random_state=Config.seed)
    meta_model.fit(X_train_meta, y_train_meta)

    # Generate Final Predictions
    final_holdout_preds = meta_model.predict(X_holdout_meta)
    final_test_preds = meta_model.predict(X_test_meta)

    # Clip probabilities to valid range [0, 1]
    final_holdout_preds = np.clip(final_holdout_preds, 0, 1)
    final_test_preds = np.clip(final_test_preds, 0, 1)

    # 4. Evaluation
    final_auc = roc_auc_score(holdout_labels, final_holdout_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(holdout_labels - final_holdout_preds)

    # Compute simple text features for correlation
    char_counts = np.array([len(t) for t in holdout_texts])
    word_counts = np.array([len(t.split()) for t in holdout_texts])
    # Handle empty strings for caps ratio
    caps_ratios = np.array(
        [
            sum(1 for c in t if c.isupper()) / len(t) if len(t) > 0 else 0.0
            for t in holdout_texts
        ]
    )

    # Calculate correlations
    corr_char = np.corrcoef(errors, char_counts)[0, 1]
    corr_word = np.corrcoef(errors, word_counts)[0, 1]
    corr_caps = np.corrcoef(errors, caps_ratios)[0, 1]

    print("Correlation between Error and Input Features:")
    print(f"  Char Count: {corr_char:.4f}")
    print(f"  Word Count: {corr_word:.4f}")
    print(f"  Caps Ratio: {corr_caps:.4f}")

    # 6. Submission
    threshold = 0.9603817733990148
    if final_auc > threshold:
        print(f"\nMetric {final_auc} > {threshold}. Generating submission...")
        submission_df = pd.read_csv("./input/sample_submission_null.csv")

        # Verify length matches
        if len(submission_df) != len(final_test_preds):
            print(
                f"Warning: Submission length {len(submission_df)} != Preds length {len(final_test_preds)}"
            )

        submission_df["Insult"] = final_test_preds
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(f"\nMetric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
