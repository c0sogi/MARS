import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, get_auc_score
from library.features import process_and_cache
from library.data import InsultDataset, get_tokenizer
from library.model import HybridDebertaModel
from library.awp import AWP
from library.engine import train_fn, eval_fn, inference_fn, get_optimizer_params


def run_training():
    # 1. Setup
    seed_everything(Config.seed)
    Config.create_directories()
    logger = get_logger()
    device = Config.device

    # Override Config for fast baseline execution if necessary
    # We stick to a reasonable number of epochs to ensure convergence but fit within time
    Config.epochs = 4

    logger.info("Starting End-to-End Soft-Target Self-Distillation Pipeline")

    # 2. Data Loading & Feature Extraction
    # Load Metadata
    train_df = pd.read_csv(Config.train_meta_path)
    val_holdout_df = pd.read_csv(Config.val_meta_path)
    test_df = pd.read_csv(Config.test_meta_path)

    # Load/Generate SVD Features
    train_svd, val_holdout_svd, test_svd = process_and_cache(load_cached_data=True)

    # Tokenizer
    tokenizer = get_tokenizer()

    # Prepare K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # ====================================================
    # Stage 1: Teacher Training
    # ====================================================
    logger.info("=== Stage 1: Teacher Training ===")

    teacher_oof_preds = np.zeros(len(train_df))

    # We iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["Insult"])
    ):
        logger.info(f"Teacher Fold {fold+1}/{Config.n_folds}")

        # Split Data
        X_train_text = train_df.iloc[train_idx]["Comment"].values
        X_train_svd = train_svd[train_idx]
        y_train = train_df.iloc[train_idx]["Insult"].values

        X_val_text = train_df.iloc[val_idx]["Comment"].values
        X_val_svd = train_svd[val_idx]
        y_val = train_df.iloc[val_idx]["Insult"].values

        # Datasets & Loaders
        train_ds = InsultDataset(X_train_text, X_train_svd, y_train, tokenizer)
        val_ds = InsultDataset(X_val_text, X_val_svd, y_val, tokenizer)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model
        model = HybridDebertaModel(pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Config.eps)

        num_train_steps = int(len(train_ds) / Config.batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # AWP
        awp = (
            AWP(
                model,
                optimizer,
                adv_lr=Config.awp_lr,
                adv_eps=Config.awp_eps,
                start_epoch=Config.awp_start_epoch,
            )
            if Config.use_awp
            else None
        )

        # Training Loop
        best_auc = 0
        best_model_path = os.path.join(
            Config.teacher_model_dir, f"teacher_fold_{fold}.bin"
        )

        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch + 1, scheduler, device, awp
            )
            val_loss, val_preds, val_labels = eval_fn(val_loader, model, device)
            auc = get_auc_score(val_labels, val_preds)

            logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # ====================================================
    # Stage 2: Soft Label Generation
    # ====================================================
    logger.info("=== Stage 2: generating Soft Labels for Test Set ===")

    test_ds = InsultDataset(
        test_df["Comment"].values, test_svd, labels=None, tokenizer=tokenizer
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_soft_labels = np.zeros(len(test_df))

    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.teacher_model_dir, f"teacher_fold_{fold}.bin")
        model = HybridDebertaModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        fold_preds = inference_fn(test_loader, model, device)
        test_soft_labels += fold_preds / Config.n_folds

        del model
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Soft labels generated.")

    # ====================================================
    # Stage 3: Student Training (Self-Distillation)
    # ====================================================
    logger.info("=== Stage 3: Student Training ===")

    # We use the same folds for the student to maintain structure,
    # but augment training data with soft-labeled test data

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["Insult"])
    ):
        logger.info(f"Student Fold {fold+1}/{Config.n_folds}")

        # Original Labeled Data
        X_train_orig = train_df.iloc[train_idx]["Comment"].values
        S_train_orig = train_svd[train_idx]
        y_train_orig = train_df.iloc[train_idx]["Insult"].values.astype(float)

        # Soft Labeled Test Data
        X_test = test_df["Comment"].values
        S_test = test_svd
        y_test = test_soft_labels  # Soft probabilities

        # Combine
        X_combined = np.concatenate([X_train_orig, X_test])
        S_combined = np.concatenate([S_train_orig, S_test])
        y_combined = np.concatenate([y_train_orig, y_test])

        # Validation Data (Original Fold Val)
        X_val = train_df.iloc[val_idx]["Comment"].values
        S_val = train_svd[val_idx]
        y_val = train_df.iloc[val_idx]["Insult"].values

        # Datasets
        train_ds = InsultDataset(X_combined, S_combined, y_combined, tokenizer)
        val_ds = InsultDataset(X_val, S_val, y_val, tokenizer)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model (Student)
        model = HybridDebertaModel(pretrained=True)
        model.to(device)

        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Config.eps)

        num_train_steps = int(len(train_ds) / Config.batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # AWP (Student also benefits)
        awp = (
            AWP(
                model,
                optimizer,
                adv_lr=Config.awp_lr,
                adv_eps=Config.awp_eps,
                start_epoch=Config.awp_start_epoch,
            )
            if Config.use_awp
            else None
        )

        best_auc = 0
        best_model_path = os.path.join(
            Config.student_model_dir, f"student_fold_{fold}.bin"
        )

        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch + 1, scheduler, device, awp
            )
            val_loss, val_preds, val_labels = eval_fn(val_loader, model, device)
            auc = get_auc_score(val_labels, val_preds)

            logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)

        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # ====================================================
    # Final Evaluation on Holdout Set
    # ====================================================
    logger.info("=== Final Evaluation on Holdout Set ===")

    val_holdout_ds = InsultDataset(
        val_holdout_df["Comment"].values,
        val_holdout_svd,
        val_holdout_df["Insult"].values,
        tokenizer,
    )
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    holdout_preds = np.zeros(len(val_holdout_df))

    # Ensemble Inference
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.student_model_dir, f"student_fold_{fold}.bin")
        model = HybridDebertaModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        fold_preds = inference_fn(val_holdout_loader, model, device)
        holdout_preds += fold_preds / Config.n_folds

        del model
        torch.cuda.empty_cache()
        gc.collect()

    final_auc = get_auc_score(val_holdout_df["Insult"].values, holdout_preds)
    print(f"Final Validation Metric: {final_auc}")

    # ====================================================
    # Failure Analysis
    # ====================================================
    logger.info("=== Failure Analysis ===")

    # Calculate errors
    y_true = val_holdout_df["Insult"].values
    errors = np.abs(y_true - holdout_preds)

    # Meta-features
    lengths = val_holdout_df["Comment"].fillna("").apply(len).values
    caps_ratio = (
        val_holdout_df["Comment"]
        .fillna("")
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x)))
        .values
    )

    # Correlations
    corr_len = np.corrcoef(errors, lengths)[0, 1]
    corr_caps = np.corrcoef(errors, caps_ratio)[0, 1]

    print("Correlation between Error and Input Features:")
    print(f"  Length: {corr_len:.4f}")
    print(f"  Caps Ratio: {corr_caps:.4f}")

    # ====================================================
    # Submission
    # ====================================================
    threshold = 0.9586453201970443
    if final_auc > threshold:
        logger.info("Validation metric met threshold. Generating submission...")

        submission_preds = np.zeros(len(test_df))

        # Reuse test loader from soft label stage
        test_ds = InsultDataset(
            test_df["Comment"].values, test_svd, labels=None, tokenizer=tokenizer
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        for fold in range(Config.n_folds):
            model_path = os.path.join(
                Config.student_model_dir, f"student_fold_{fold}.bin"
            )
            model = HybridDebertaModel(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            fold_preds = inference_fn(test_loader, model, device)
            submission_preds += fold_preds / Config.n_folds

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Save submission
        submission_df = pd.DataFrame(
            {
                "id": range(
                    len(submission_preds)
                ),  # Assuming index based ID if not present
                "prediction": submission_preds,
            }
        )

        # The sample submission format in description: | Insult | Date | Comment | (Wait, sample_submission_null.csv has 3 cols)
        # The task description says: "Your predictions should be a number in the range [0,1]. See 'sample_submissions_null.csv' for the correct format."
        # The sample submission usually implies we need to match the rows.
        # Let's check sample_submission_null.csv structure from description.
        # It has columns: Insult, Date, Comment.
        # Usually for this competition type, we just need to fill the 'Insult' column with probabilities.

        sample_sub = pd.read_csv("./input/sample_submission_null.csv")
        sample_sub["Insult"] = submission_preds
        sample_sub.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")

    else:
        logger.info(
            f"Validation metric {final_auc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run_training()
