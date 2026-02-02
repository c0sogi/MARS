import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from the provided library files
from library.config import CFG
from library.dataset import prepare_data, PhraseDataset
from library.model import PhraseModel
from library.loss import HybridLoss
from library.optimization import get_optimizer_params, AWP, EMA
from library.engine import train_fn, valid_fn, inference_fn
from library.utils import seed_everything, get_score, get_logger


def run():
    # ------------------------------------------------------------------------
    # 1. Setup and Configuration Adjustments for Fast Baseline
    # ------------------------------------------------------------------------
    seed_everything(CFG.seed)

    # Create submission directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Adjust CFG for speed (Fast Baseline requirements)
    CFG.epochs = 2
    CFG.trn_fold = [0, 1]  # Train only 2 folds to save time
    CFG.max_len = 130  # Reduce sequence length slightly for speed
    CFG.batch_scheduler = True

    logger = get_logger(os.path.join(CFG.output_dir, "train.log"))
    logger.info(
        f"Configuration: Epochs={CFG.epochs}, Folds={CFG.trn_fold}, Device={CFG.device}"
    )

    # ------------------------------------------------------------------------
    # 2. Data Loading & Preparation
    # ------------------------------------------------------------------------
    logger.info("Loading and preparing data...")

    # Load metadata and inject context text
    train_df = prepare_data(CFG.train_path, load_cached_data=True)
    val_df = prepare_data(CFG.val_path, load_cached_data=True)
    test_df = prepare_data(CFG.test_path, load_cached_data=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # ------------------------------------------------------------------------
    # 3. Stratified K-Fold Training Loop
    # ------------------------------------------------------------------------
    # We split the 'train_df' further into folds for CV training
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    # Create 'fold' column
    train_df["fold"] = -1
    # Stratify by score (converted to string to treat as class for stratification)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["score"].astype(str))
    ):
        train_df.loc[val_idx, "fold"] = fold

    # Store fold models for inference later
    fold_model_paths = []

    for fold in CFG.trn_fold:
        logger.info(f"=== Starting Training Fold {fold} ===")

        # Split Data
        fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = PhraseDataset(fold_train_df, tokenizer, max_len=CFG.max_len)
        valid_dataset = PhraseDataset(fold_val_df, tokenizer, max_len=CFG.max_len)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Model
        model = PhraseModel(CFG.model_name, pretrained=True)
        model.to(CFG.device)

        # Optimizer (LLRD)
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=CFG.encoder_lr,
            head_lr=CFG.head_lr,
            weight_decay=CFG.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=1e-6)

        # Scheduler
        num_train_steps = int(len(fold_train_df) / CFG.train_batch_size * CFG.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
            num_training_steps=num_train_steps,
            num_cycles=CFG.num_cycles,
        )

        # Loss
        criterion = HybridLoss()

        # Advanced Strategies: AWP & EMA
        awp = (
            AWP(model, optimizer, adv_lr=CFG.awp_lr, adv_eps=CFG.awp_eps)
            if CFG.awp
            else None
        )
        ema = EMA(model, decay=CFG.ema_decay) if CFG.ema else None

        # Training Loop
        best_score = -1.0
        best_model_path = os.path.join(CFG.output_dir, f"model_fold{fold}.pth")

        for epoch in range(CFG.epochs):
            # Train
            avg_loss = train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                CFG.device,
                awp,
                ema,
            )

            # Validation (using EMA weights)
            avg_val_loss, score, _ = valid_fn(
                valid_loader, model, criterion, CFG.device, ema
            )

            logger.info(
                f"Fold {fold} | Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Pearson: {score:.4f}"
            )

            # Save Best Model
            if score > best_score:
                best_score = score
                logger.info(f"Score Improved. Saving model to {best_model_path}")
                # Save EMA weights if available, else model weights
                if ema is not None:
                    ema.apply_shadow()
                    torch.save(model.state_dict(), best_model_path)
                    ema.restore()
                else:
                    torch.save(model.state_dict(), best_model_path)

        fold_model_paths.append(best_model_path)

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        torch.cuda.empty_cache()
        logger.info(f"Fold {fold} finished.")

    # ------------------------------------------------------------------------
    # 4. Final Validation on Hold-Out Set (Ensemble)
    # ------------------------------------------------------------------------
    logger.info("=== Performing Final Validation on Hold-Out Set ===")

    val_dataset = PhraseDataset(val_df, tokenizer, max_len=CFG.max_len)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Ensemble Predictions
    val_preds = []

    for model_path in fold_model_paths:
        logger.info(f"Loading model from {model_path} for inference...")
        model = PhraseModel(CFG.model_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        model.to(CFG.device)

        preds = inference_fn(
            val_loader, model, CFG.device, ema=None
        )  # EMA already applied during save
        val_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_val_preds = np.mean(val_preds, axis=0)

    # Calculate Metric
    final_metric = get_score(val_df["score"].values, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ------------------------------------------------------------------------
    # 5. Failure Analysis
    # ------------------------------------------------------------------------
    logger.info("=== Running Failure Analysis ===")

    val_df["pred"] = avg_val_preds
    val_df["abs_error"] = (val_df["score"] - val_df["pred"]).abs()

    # Feature Engineering for Analysis
    val_df["anchor_len"] = val_df["anchor"].astype(str).apply(len)
    val_df["target_len"] = val_df["target"].astype(str).apply(len)
    val_df["context_len"] = val_df["context_text"].astype(str).apply(len)

    def get_common_word_count(row):
        s1 = set(str(row["anchor"]).lower().split())
        s2 = set(str(row["target"]).lower().split())
        return len(s1.intersection(s2))

    val_df["common_words"] = val_df.apply(get_common_word_count, axis=1)

    # Correlation Analysis
    analysis_cols = [
        "anchor_len",
        "target_len",
        "context_len",
        "common_words",
        "abs_error",
    ]
    correlations = (
        val_df[analysis_cols].corr()["abs_error"].sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ------------------------------------------------------------------------
    # 6. Submission
    # ------------------------------------------------------------------------
    THRESHOLD = 0.8698034882545471

    if final_metric > THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.5f} > {THRESHOLD}. Generating submission..."
        )

        test_dataset = PhraseDataset(test_df, tokenizer, max_len=CFG.max_len)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        test_preds = []

        for model_path in fold_model_paths:
            model = PhraseModel(CFG.model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=CFG.device))
            model.to(CFG.device)

            preds = inference_fn(test_loader, model, CFG.device, ema=None)
            test_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_preds, axis=0)

        # Create submission dataframe
        submission = pd.DataFrame({"id": test_df["id"], "score": avg_test_preds})

        save_path = os.path.join(submission_dir, "submission.csv")
        submission.to_csv(save_path, index=False)
        logger.info(f"Submission saved to {save_path}")

    else:
        logger.warning(
            f"Validation metric {final_metric:.5f} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
