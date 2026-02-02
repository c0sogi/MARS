import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from library.configuration import Config, seed_everything
from library.utilities import get_logger, compute_metrics
from library.dataset import load_supervised_data, get_tokenizer, Collate, EssayDataset
from library.modeling import EssayModel, get_optimizer_params

logger = get_logger("Training")


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        with autocast():
            logits = model(input_ids, attention_mask)
            # Logits shape: (batch_size, 1), Labels shape: (batch_size)
            loss = criterion(logits.view(-1), labels)

            # Scale loss for gradient accumulation
            if Config.GRAD_ACCUM_STEPS > 1:
                loss = loss / Config.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * Config.GRAD_ACCUM_STEPS * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_fn(model, dataloader, device):
    """
    Executes validation loop. Returns average loss, predictions, and metrics.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1), labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.append(logits.view(-1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    predictions = np.concatenate(preds)
    true_labels = np.concatenate(targets)

    metrics = compute_metrics(true_labels, predictions)

    return epoch_loss, predictions, true_labels, metrics


def run_training(debug: bool = False, load_cached_data: bool = True):
    """
    Main function to run the 5-Fold Cross-Validation Training.
    """
    seed_everything(Config.SEED)

    # Create working directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Load Data
    tokenizer = get_tokenizer()
    # Load full training data
    full_dataset = load_supervised_data(
        "train", tokenizer, load_cached_data=load_cached_data, debug=debug
    )
    df = full_dataset.df

    # 2. Prepare Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Placeholder for Out-of-Fold predictions
    oof_df = pd.DataFrame()

    # Determine Pre-trained Checkpoint Path (Stage 1)
    # Check if MLM checkpoint exists and has config
    mlm_path = Config.MLM_CHECKPOINT_DIR
    has_mlm = os.path.exists(os.path.join(mlm_path, "config.json"))
    checkpoint_path = mlm_path if has_mlm else None

    if has_mlm:
        logger.info(f"MLM Checkpoint found. Using backbone from: {mlm_path}")
    else:
        logger.info(
            f"No MLM Checkpoint found. Using base backbone: {Config.MODEL_BACKBONE}"
        )

    # 3. Training Loop per Fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["score"])):
        logger.info(f"\n{'='*20} Fold {fold+1} / {Config.NUM_FOLDS} {'='*20}")

        # Split Data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = EssayDataset(train_df)
        val_ds = EssayDataset(val_df)

        collate_fn = Collate(tokenizer)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Initialize Model
        model = EssayModel(checkpoint_path=checkpoint_path, pretrained=True)
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer_parameters = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=Config.LEARNING_RATE, eps=1e-6, betas=(0.9, 0.999)
        )

        num_train_steps = int(
            len(train_df)
            / Config.TRAIN_BATCH_SIZE
            / Config.GRAD_ACCUM_STEPS
            * Config.EPOCHS
        )
        num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        scaler = GradScaler()

        # Tracking
        best_qwk = -1.0
        best_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 2  # Strict early stopping for time limit

        save_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        # Epoch Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(
                model, train_loader, optimizer, scheduler, Config.DEVICE, scaler, epoch
            )
            val_loss, val_preds, val_labels, val_metrics = valid_fn(
                model, val_loader, Config.DEVICE
            )

            val_qwk = val_metrics["qwk"]

            logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val QWK: {val_qwk:.6f}"
            )

            # Save Best Model (Prioritize QWK)
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                best_loss = val_loss
                logger.info(f"New best QWK: {best_qwk:.6f}. Saving model...")
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        # Load Best Model for OOF
        logger.info(f"Loading best model for Fold {fold+1} OOF generation...")
        model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))

        _, oof_preds, _, _ = valid_fn(model, val_loader, Config.DEVICE)

        # Store OOF
        val_df["pred_score"] = oof_preds
        oof_df = pd.concat([oof_df, val_df], axis=0)

        # Cleanup
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Save OOF Predictions
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    # Calculate Overall CV Score
    overall_metrics = compute_metrics(
        oof_df["score"].values, oof_df["pred_score"].values
    )
    logger.info(f"Overall CV QWK: {overall_metrics['qwk']:.6f}")

    return oof_df
