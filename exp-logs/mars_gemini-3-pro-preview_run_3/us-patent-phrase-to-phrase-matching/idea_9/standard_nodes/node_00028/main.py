import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import load_data, PhraseDataset
from library.model import PhraseModel
from library.engine import train_fn, valid_fn, EMA

# Suppress transformers logging
logging.set_verbosity_error()


def run():
    # 1. Setup
    logger = get_logger()
    seed_everything(Config.seed)

    # Override Config for Fast Baseline execution
    # 2 epochs ensures completion within 2 hours for 5 folds on A100
    Config.epochs = 2

    logger.info(f"Starting run with device: {Config.device}")

    # 2. Data Loading
    # Load Train (for CV) and Val (Hold-out)
    train_df = load_data("train", load_cached_data=True)
    val_holdout_df = load_data("val", load_cached_data=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Stratified K-Fold
    # Stratify by score to maintain distribution
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    fold_splits = list(skf.split(train_df, train_df["score"].astype(str)))

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        logger.info(f"========== Fold {fold+1}/{Config.n_folds} ==========")

        # Split Data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = PhraseDataset(fold_train_df, tokenizer, Config.max_len)
        valid_dataset = PhraseDataset(fold_val_df, tokenizer, Config.max_len)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = PhraseModel()
        model.to(Config.device)

        # Optimizer & Scheduler (LLRD)
        optimizer_params = model.get_optimizer_params(
            encoder_lr=Config.learning_rate,
            decoder_lr=Config.head_lr,
            weight_decay=Config.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_params)

        num_train_steps = int(
            len(fold_train_df) / Config.train_batch_size * Config.epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # EMA
        ema = None
        if Config.use_ema:
            ema = EMA(model, Config.ema_decay)
            ema.register()

        # Training
        best_pearson = -1.0
        best_model_state = None

        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, scheduler, Config.device, epoch, ema
            )

            # Validation
            val_loss, val_pearson, _ = valid_fn(valid_loader, model, Config.device, ema)

            logger.info(
                f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_pearson:.4f}"
            )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                # Save best model state (using EMA weights if available)
                if ema:
                    ema.apply_shadow()
                    best_model_state = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }
                    ema.restore()
                else:
                    best_model_state = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }

        # Save Fold Model
        save_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")
        torch.save(best_model_state, save_path)
        logger.info(f"Fold {fold+1} Best Pearson: {best_pearson:.4f}. Model saved.")

        # Clean up
        del (
            model,
            optimizer,
            scheduler,
            ema,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        torch.cuda.empty_cache()

    # 5. Hold-out Validation (Ensemble)
    logger.info("========== Hold-out Validation (Ensemble) ==========")

    # Load Hold-out Dataset
    holdout_dataset = PhraseDataset(val_holdout_df, tokenizer, Config.max_len)
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Ensemble Predictions
    ensemble_preds = np.zeros(len(val_holdout_df))

    for fold in range(Config.n_folds):
        logger.info(f"Inference with Fold {fold+1} model...")
        model = PhraseModel()
        model.load_state_dict(
            torch.load(os.path.join(Config.working_dir, f"model_fold_{fold}.pth"))
        )
        model.to(Config.device)
        model.eval()

        _, _, preds = valid_fn(holdout_loader, model, Config.device)
        ensemble_preds += preds

        del model
        torch.cuda.empty_cache()

    ensemble_preds /= Config.n_folds
    ensemble_preds = np.clip(ensemble_preds, 0, 1)

    # Calculate Metric
    holdout_labels = val_holdout_df["score"].values
    final_pearson = np.corrcoef(ensemble_preds, holdout_labels)[0, 1]

    print(f"Final Validation Metric: {final_pearson}")

    # 6. Failure Analysis
    logger.info("========== Failure Analysis ==========")
    errors = np.abs(ensemble_preds - holdout_labels)

    # Compute features
    anchor_lens = val_holdout_df["anchor"].astype(str).apply(len).values
    target_lens = val_holdout_df["target"].astype(str).apply(len).values
    context_counts = val_holdout_df["context"].value_counts().to_dict()
    context_freqs = val_holdout_df["context"].map(context_counts).values

    # Correlation
    corr_anchor = np.corrcoef(errors, anchor_lens)[0, 1]
    corr_target = np.corrcoef(errors, target_lens)[0, 1]
    corr_context = np.corrcoef(errors, context_freqs)[0, 1]

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Anchor Length: {corr_anchor:.4f}")
    print(f"  Target Length: {corr_target:.4f}")
    print(f"  Context Freq:  {corr_context:.4f}")

    # 7. Submission
    THRESHOLD = 0.8698034882545471

    if final_pearson > THRESHOLD:
        logger.info("Metric passed threshold. Generating submission...")

        test_df = load_data("test", load_cached_data=True)
        test_dataset = PhraseDataset(test_df, tokenizer, Config.max_len)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds = np.zeros(len(test_df))

        for fold in range(Config.n_folds):
            logger.info(f"Test Inference with Fold {fold+1} model...")
            model = PhraseModel()
            model.load_state_dict(
                torch.load(os.path.join(Config.working_dir, f"model_fold_{fold}.pth"))
            )
            model.to(Config.device)
            model.eval()

            _, _, preds = valid_fn(test_loader, model, Config.device)
            test_preds += preds

            del model
            torch.cuda.empty_cache()

        test_preds /= Config.n_folds
        test_preds = np.clip(test_preds, 0, 1)

        # Create Submission File
        submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")

    else:
        logger.info(
            f"Metric {final_pearson} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
