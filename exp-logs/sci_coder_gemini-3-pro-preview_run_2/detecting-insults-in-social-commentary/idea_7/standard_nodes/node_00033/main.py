import os
import sys
import copy
import gc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library import utils, data, model, engine, awp

# Suppress HF warnings
hf_logging.set_verbosity_error()


def run_tapt(df_train, df_val, df_test):
    """
    Stage 1: Task-Adaptive Pre-Training (MLM)
    """
    utils.get_logger().info("=== Stage 1: TAPT ===")

    # Prepare text
    texts = data.prepare_tapt_data(df_train, df_val, df_test)
    tokenizer = data.get_tokenizer()

    # Dataset & Loader
    ds = data.MLMDataset(texts, tokenizer, Config.max_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )

    loader = DataLoader(
        ds,
        batch_size=Config.tapt_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Model
    model_tapt = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model_tapt.to(Config.device)

    # Optimization
    optimizer = AdamW(
        model_tapt.parameters(),
        lr=Config.tapt_lr,
        weight_decay=Config.tapt_weight_decay,
    )
    num_train_steps = len(loader) * Config.tapt_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Initialize Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    # Train
    for epoch in range(Config.tapt_epochs):
        loss = engine.train_mlm(
            model_tapt, loader, optimizer, scheduler, Config.device, epoch, scaler
        )
        utils.get_logger().info(f"TAPT Epoch {epoch+1} Loss: {loss:.4f}")

    # Save
    model_tapt.save_pretrained(Config.tapt_output_dir)
    tokenizer.save_pretrained(Config.tapt_output_dir)
    utils.get_logger().info(f"TAPT model saved to {Config.tapt_output_dir}")

    # Cleanup
    del model_tapt, optimizer, scheduler, loader, ds
    torch.cuda.empty_cache()
    gc.collect()


def run_teacher(df_train, df_test):
    """
    Stage 2: Teacher Training (5-Fold)
    Returns:
        test_preds: Average predictions on test set
    """
    utils.get_logger().info("=== Stage 2: Teacher Training ===")

    # Inputs
    X = df_train
    y = df_train[Config.target_col].values

    # Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Storage
    test_preds_fold = []

    tokenizer = data.get_tokenizer()

    # Prepare Test Loader once
    test_ds = data.InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.teacher_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        utils.get_logger().info(f"--- Teacher Fold {fold+1}/{Config.n_folds} ---")

        # Split
        df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = data.InsultDataset(df_fold_train, tokenizer, Config.max_len)
        val_ds = data.InsultDataset(df_fold_val, tokenizer, Config.max_len)

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.teacher_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.teacher_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        # Model (Load from TAPT)
        teacher_model = model.InsultModel(
            model_path=Config.tapt_output_dir, pretrained=True
        )
        teacher_model.to(Config.device)

        # Optimizer
        optimizer = AdamW(
            teacher_model.parameters(),
            lr=Config.teacher_lr,
            weight_decay=Config.teacher_weight_decay,
        )
        num_train_steps = len(train_loader) * Config.teacher_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Train
        save_path = os.path.join(Config.teacher_output_dir, f"model_fold_{fold}.pth")
        teacher_model, best_auc = engine.run_training(
            teacher_model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            Config.device,
            Config.teacher_epochs,
            Config.teacher_patience,
            save_path,
            use_awp=False,
        )

        # Inference on Test
        preds = engine.inference_fn(teacher_model, test_loader, Config.device)
        test_preds_fold.append(preds)

        # Cleanup
        del (
            teacher_model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_ds,
            val_ds,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions
    avg_test_preds = np.mean(test_preds_fold, axis=0)

    return avg_test_preds


def run_student(df_train_orig, df_augmented, df_val_holdout, df_test):
    """
    Stage 3: Student Training (5-Fold on Augmented Data with AWP)
    """
    utils.get_logger().info("=== Stage 3: Student Training ===")

    X = df_train_orig
    y = df_train_orig[Config.target_col].values

    # Identify pseudo-labeled data
    # df_augmented is [df_train_orig; df_pseudo]
    n_train = len(df_train_orig)
    df_pseudo = df_augmented.iloc[n_train:].copy()

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    tokenizer = data.get_tokenizer()

    # Loaders for Holdout and Test
    holdout_ds = data.InsultDataset(df_val_holdout, tokenizer, Config.max_len)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=Config.student_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    test_ds = data.InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.student_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    student_test_preds = []
    student_holdout_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        utils.get_logger().info(f"--- Student Fold {fold+1}/{Config.n_folds} ---")

        # Original Train/Val for this fold
        df_fold_train_orig = df_train_orig.iloc[train_idx]
        df_fold_val = df_train_orig.iloc[val_idx].reset_index(drop=True)

        # Combine Fold Train with Pseudo Data
        df_fold_train_final = pd.concat([df_fold_train_orig, df_pseudo]).reset_index(
            drop=True
        )

        # Datasets
        train_ds = data.InsultDataset(df_fold_train_final, tokenizer, Config.max_len)
        val_ds = data.InsultDataset(df_fold_val, tokenizer, Config.max_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.student_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.student_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        # Model (Load from TAPT)
        student_model = model.InsultModel(
            model_path=Config.tapt_output_dir, pretrained=True
        )
        student_model.to(Config.device)

        # Optimizer
        optimizer = AdamW(
            student_model.parameters(),
            lr=Config.student_lr,
            weight_decay=Config.student_weight_decay,
        )
        num_train_steps = len(train_loader) * Config.student_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Train with AWP
        save_path = os.path.join(Config.student_output_dir, f"model_fold_{fold}.pth")
        student_model, best_auc = engine.run_training(
            student_model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            Config.device,
            Config.student_epochs,
            100,  # Loose patience for student
            save_path,
            use_awp=Config.use_awp,
        )

        # Inference
        # 1. Test
        t_preds = engine.inference_fn(student_model, test_loader, Config.device)
        student_test_preds.append(t_preds)

        # 2. Holdout Validation
        h_preds = engine.inference_fn(student_model, holdout_loader, Config.device)
        student_holdout_preds.append(h_preds)

        # Cleanup
        del (
            student_model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_ds,
            val_ds,
        )
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(student_test_preds, axis=0), np.mean(student_holdout_preds, axis=0)


def main():
    utils.set_seed(Config.seed)

    # 1. Data Loading
    df_train, df_val, df_test = data.prepare_supervised_data(load_cached_data=True)

    # 2. TAPT
    run_tapt(df_train, df_val, df_test)

    # 3. Teacher
    teacher_test_preds = run_teacher(df_train, df_test)

    # 4. Pseudo Labeling
    df_augmented = data.prepare_pseudo_data(df_train, df_test, teacher_test_preds)

    # 5. Student
    student_test_preds, student_holdout_preds = run_student(
        df_train, df_augmented, df_val, df_test
    )

    # 6. Evaluation
    targets = df_val[Config.target_col].values
    final_auc = roc_auc_score(targets, student_holdout_preds)

    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(targets - student_holdout_preds)
    texts = df_val[Config.input_col].astype(str).tolist()
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]

    corr_char = np.corrcoef(errors, char_lens)[0, 1]
    corr_word = np.corrcoef(errors, word_lens)[0, 1]

    print("Failure Analysis:")
    print(f"Correlation (Error vs Char Length): {corr_char:.4f}")
    print(f"Correlation (Error vs Word Length): {corr_word:.4f}")

    # 7. Submission
    threshold = 0.9632101806239738
    if final_auc > threshold:
        utils.get_logger().info(
            "Validation metric met threshold. Generating submission."
        )
        submission = pd.read_csv(
            os.path.join(Config.input_dir, "sample_submission_null.csv")
        )

        # Handle potential length mismatch if test.csv has issues (robustness)
        if len(submission) != len(student_test_preds):
            # Just in case, though metadata/test.csv should match sample_submission
            submission = submission.iloc[: len(student_test_preds)]

        submission[Config.target_col] = student_test_preds
        submission.to_csv(Config.submission_path, index=False)
        utils.get_logger().info(f"Submission saved to {Config.submission_path}")
    else:
        utils.get_logger().info(
            f"Validation metric {final_auc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
