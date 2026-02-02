import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

# Import from library
from library.config import Config
from library.utils import seed_everything, get_score, Logger, save_checkpoint
from library.data import load_processed_data, InsultDataset, clean_text
from library.model import InsultModel
from library.engine import (
    run_tapt,
    train_fn,
    valid_fn,
    inference_fn,
    get_optimizer_params,
)
from library.awp import AWP


def main():
    # 1. Configuration & Setup
    # Initialize config with settings optimized for the 54-minute time limit
    # Using 3 epochs for supervised training and 2 for TAPT ensures completion while testing the idea.
    # Cite debug_lesson_3: Disable gradient checkpointing and reduce batch size to fix RuntimeError and avoid OOM.
    config = Config(epochs=3, train_batch_size=4)
    config.tapt_epochs = 2
    config.gradient_checkpointing = False

    seed_everything(config.seed)
    logger = Logger(os.path.join(config.output_dir, "train_log.txt"))
    logger.log("Starting execution...")

    # 2. Data Loading
    # Load processed dataframes (train, val, test)
    train_df, val_df, test_df = load_processed_data(config, load_cached_data=True)

    # 3. Task-Adaptive Pre-Training (TAPT)
    # Adapts the backbone to the specific domain language using all available text
    run_tapt(config, logger)

    # 4. Cross-Validation Training
    skf = StratifiedKFold(
        n_splits=config.num_folds, shuffle=True, random_state=config.seed
    )

    # We use the training metadata for CV
    X = train_df
    y = train_df[config.target_col].values

    # Determine tokenizer path (TAPT output if available, else base model)
    tokenizer_path = (
        config.tapt_output_dir
        if config.use_tapt and os.path.exists(config.tapt_output_dir)
        else config.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    best_auc_scores = []

    for fold, (train_idx, dev_idx) in enumerate(skf.split(X, y)):
        logger.log(f"\n{'='*20} Fold {fold+1} / {config.num_folds} {'='*20}")

        # Split Data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_dev_df = train_df.iloc[dev_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = InsultDataset(fold_train_df, tokenizer, config.max_length)
        dev_dataset = InsultDataset(fold_dev_df, tokenizer, config.max_length)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        # InsultModel logic handles loading from TAPT path automatically
        model = InsultModel(config, pretrained=True)
        model.to(config.device)

        # Optimizer (with Layer-wise Learning Rate Decay)
        optimizer_grouped_parameters = get_optimizer_params(model, config)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-6
        )

        # Scheduler
        num_train_steps = int(
            len(fold_train_df) / config.train_batch_size * config.epochs
        )
        num_warmup_steps = int(num_train_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Adversarial Weight Perturbation (AWP)
        awp = AWP(
            model,
            optimizer,
            adv_lr=config.awp_lr,
            adv_eps=config.awp_eps,
            start_epoch=config.awp_start_epoch,
        )

        # Training Loop
        best_fold_auc = 0

        for epoch in range(config.epochs):
            avg_loss = train_fn(
                train_loader,
                model,
                optimizer,
                config.device,
                scheduler,
                epoch,
                config,
                awp,
            )
            val_loss, val_preds = valid_fn(dev_loader, model, config.device)
            val_labels = fold_dev_df[config.target_col].values
            val_auc = get_score(val_labels, val_preds)

            logger.log(
                f"Epoch {epoch+1} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f} - val_auc: {val_auc:.4f}"
            )

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                save_path = os.path.join(config.output_dir, f"model_fold_{fold}.pth")
                torch.save(model.state_dict(), save_path)

        best_auc_scores.append(best_fold_auc)
        logger.log(f"Fold {fold+1} Best AUC: {best_fold_auc:.4f}")

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, awp, train_loader, dev_loader
        torch.cuda.empty_cache()

    logger.log(f"Average CV AUC: {np.mean(best_auc_scores):.4f}")

    # 5. Hold-out Validation & Failure Analysis
    logger.log("\nRunning Hold-out Validation...")

    val_dataset = InsultDataset(val_df, tokenizer, config.max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Ensemble Prediction on Hold-out Validation Set
    fold_preds = []
    for fold in range(config.num_folds):
        model = InsultModel(config, pretrained=False)
        model.load_state_dict(
            torch.load(os.path.join(config.output_dir, f"model_fold_{fold}.pth"))
        )
        model.to(config.device)

        preds = inference_fn(val_loader, model, config.device)
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    avg_val_preds = np.mean(fold_preds, axis=0)
    final_val_auc = get_score(val_df[config.target_col].values, avg_val_preds)

    # Required Output
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    logger.log("Performing Failure Analysis...")
    val_df["pred"] = avg_val_preds
    val_df["error"] = np.abs(val_df[config.target_col] - val_df["pred"])
    val_df["text_len"] = val_df["Comment"].apply(lambda x: len(str(x)))
    val_df["has_date"] = val_df["Date"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
    )

    # Calculate correlations
    corr_len = np.corrcoef(val_df["error"], val_df["text_len"])[0, 1]
    corr_date = np.corrcoef(val_df["error"], val_df["has_date"])[0, 1]

    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Has Date): {corr_date:.4f}")

    # 6. Submission
    threshold = 0.9632101806239738
    if final_val_auc > threshold:
        logger.log("Generating Submission...")
        test_dataset = InsultDataset(
            test_df, tokenizer, config.max_length, is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        test_fold_preds = []
        for fold in range(config.num_folds):
            model = InsultModel(config, pretrained=False)
            model.load_state_dict(
                torch.load(os.path.join(config.output_dir, f"model_fold_{fold}.pth"))
            )
            model.to(config.device)

            preds = inference_fn(test_loader, model, config.device)
            test_fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_fold_preds, axis=0)

        # Load sample submission and fill predictions
        sub_df = pd.read_csv(config.sample_submission_path)
        sub_df["Insult"] = avg_test_preds
        sub_df.to_csv(config.submission_path, index=False)
        logger.log(f"Submission saved to {config.submission_path}")
    else:
        logger.log(
            f"Validation AUC {final_val_auc} did not beat threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
