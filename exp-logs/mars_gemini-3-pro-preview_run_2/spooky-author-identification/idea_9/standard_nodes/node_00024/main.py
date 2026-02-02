import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.data_factory import load_data, get_classical_features, TextDataset
from library.classical_engine import run_classical_cv
from library.neural_model import CustomDeberta
from library.optimization import get_optimizer_grouped_parameters
from library.engine import train_fn, eval_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    # 1. Setup & Configuration Overrides for Speed
    seed_everything(Config.seed)

    # Optimization: Reduce folds and epochs to ensure execution within 1 hour
    # 3 folds guarantees full OOF coverage for stacking while saving 40% time vs 5 folds.
    Config.n_folds = 3
    Config.epochs = 2

    print(
        f"Configuration: Folds={Config.n_folds}, Epochs={Config.epochs}, Device={Config.device}"
    )

    # 2. Data Loading
    print("Loading data...")
    train_df_meta, val_df_meta, test_df_meta = load_data()

    # Combine Train and Val for robust Cross-Validation Stacking
    # This aligns with the classical_engine logic
    full_df = pd.concat([train_df_meta, val_df_meta]).reset_index(drop=True)
    y_full = full_df["author"].map(Config.label2id).values

    print(f"Full Training Data Shape: {full_df.shape}")
    print(f"Test Data Shape: {test_df_meta.shape}")

    # 3. Classical Branch
    print("\n=== Running Classical Branch ===")
    # Generate Features
    data_dict = get_classical_features(
        train_df_meta, val_df_meta, test_df_meta, load_cached_data=True
    )

    # Run Models (LR, NB, XGB)
    # Note: run_classical_cv handles the concatenation of train/val internally to match our full_df
    classical_oof, classical_test = run_classical_cv(
        data_dict, train_df_meta, val_df_meta, load_cached_preds=True
    )

    # 4. Neural Branch
    print("\n=== Running Neural Branch ===")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Prepare Test Loader (Once)
    test_ds = TextDataset(test_df_meta, tokenizer, Config.max_len, is_test=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Storage for Neural Predictions
    neural_oof = np.zeros((len(full_df), Config.num_classes))
    neural_test_preds = np.zeros((len(test_df_meta), Config.num_classes))

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y_full)):
        print(f"\n--- Neural Fold {fold + 1}/{Config.n_folds} ---")

        # Prepare Fold Data
        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)

        train_ds = TextDataset(fold_train_df, tokenizer, Config.max_len)
        val_ds = TextDataset(fold_val_df, tokenizer, Config.max_len)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomDeberta(Config.model_name)
        model.to(Config.device)

        # Optimizer & Scheduler
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=Config.lr, eps=1e-6
        )

        num_train_steps = int(
            len(fold_train_df)
            / Config.train_batch_size
            / Config.gradient_accumulation_steps
            * Config.epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Training Loop
        best_val_loss = np.inf
        best_val_preds = None

        for epoch in range(Config.epochs):
            _ = train_fn(
                train_loader, model, optimizer, scheduler, epoch, Config.device
            )
            val_loss, val_preds = eval_fn(val_loader, model, Config.device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_preds = val_preds

        # Store Best OOF Predictions
        neural_oof[val_idx] = best_val_preds

        # Inference on Test Set
        fold_test_preds = inference_fn(test_loader, model, Config.device)
        neural_test_preds += fold_test_preds / Config.n_folds

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 5. Stacking (Meta-Learner)
    print("\n=== Stacking (Late Fusion) ===")

    # Construct Feature Matrix: [LR, NB, XGB, Neural]
    # Classical OOFs are dictionaries aligned with full_df
    X_stack_train = np.hstack(
        [classical_oof["lr"], classical_oof["nb"], classical_oof["xgb"], neural_oof]
    )

    X_stack_test = np.hstack(
        [
            classical_test["lr"],
            classical_test["nb"],
            classical_test["xgb"],
            neural_test_preds,
        ]
    )

    print(f"Stacking Train Shape: {X_stack_train.shape}")

    # Train Meta-Learner
    meta_model = LogisticRegression(C=1.0, random_state=Config.seed, solver="lbfgs")
    meta_model.fit(X_stack_train, y_full)

    # Generate Final Probabilities
    final_oof_probs = meta_model.predict_proba(X_stack_train)
    final_test_probs = meta_model.predict_proba(X_stack_test)

    # 6. Validation Assessment
    # We must report the metric on the specific hold-out validation set (val_df).
    # Since we concatenated [train, val], the validation set corresponds to the last len(val_df) rows.
    val_size = len(val_df_meta)
    val_indices = np.arange(len(full_df) - val_size, len(full_df))

    val_preds_subset = final_oof_probs[val_indices]
    val_labels_subset = y_full[val_indices]

    final_score = get_score(val_labels_subset, val_preds_subset)
    print(f"Final Validation Metric: {final_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Log Loss contribution per sample: -log(p_true)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    p_true = val_preds_subset[np.arange(len(val_labels_subset)), val_labels_subset]
    p_true = np.clip(p_true, epsilon, 1 - epsilon)
    sample_losses = -np.log(p_true)

    # Compute Text Lengths for Correlation
    val_df_meta["char_count"] = val_df_meta["text"].astype(str).apply(len)
    val_df_meta["word_count"] = (
        val_df_meta["text"].astype(str).apply(lambda x: len(x.split()))
    )

    corr_char = np.corrcoef(sample_losses, val_df_meta["char_count"])[0, 1]
    corr_word = np.corrcoef(sample_losses, val_df_meta["word_count"])[0, 1]

    print(f"Correlation (Error vs Char Count): {corr_char}")
    print(f"Correlation (Error vs Word Count): {corr_word}")

    # 8. Submission
    threshold = 0.23237805822413304
    if final_score < threshold:
        print("\nScore meets threshold. Generating submission...")

        # Load sample submission to ensure correct format and ID order
        submission = pd.read_csv("./input/sample_submission.csv")

        # Assign probabilities (Order: EAP, HPL, MWS matches Config.id2label 0, 1, 2)
        submission["EAP"] = final_test_probs[:, 0]
        submission["HPL"] = final_test_probs[:, 1]
        submission["MWS"] = final_test_probs[:, 2]

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nScore {final_score} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
