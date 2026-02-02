import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, DataCollatorWithPadding

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    AverageMeter,
    calculate_roc_auc,
    EarlyStopping,
    save_npy,
)
from library.data_processing import load_raw_data, ToxicityDataset
from library.models import CustomTransformer
from library.training_utils import get_llrd_optimizer_params, get_scheduler


def eval_fn(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the ROC AUC score.
    """
    model.eval()
    loss_meter = AverageMeter()
    final_targets = []
    final_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            # Mixed precision is not strictly necessary for eval but can speed it up
            with autocast(enabled=True):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            loss_meter.update(loss.item(), input_ids.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            final_targets.append(labels.cpu().numpy())
            final_preds.append(preds.cpu().numpy())

    final_targets = np.vstack(final_targets)
    final_preds = np.vstack(final_preds)

    # Calculate Metric
    auc_score = calculate_roc_auc(final_targets, final_preds)

    return loss_meter.avg, auc_score


def inference_fn(model, dataloader, device):
    """
    Generates probability predictions for a given dataloader.
    Returns a numpy array of shape (N, num_classes).
    """
    model.eval()
    final_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=True):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.sigmoid(logits)
            final_preds.append(preds.cpu().numpy())

    return np.vstack(final_preds)


def run_supervised_training(
    model_name: str,
    pretrained_path: str = None,
    save_model_path: str = None,
    val_preds_save_path: str = None,
    test_preds_save_path: str = None,
    debug: bool = Config.DEBUG,
):
    """
    Orchestrates the supervised fine-tuning loop for a single Transformer model.

    Args:
        model_name: Name of the backbone (e.g., 'roberta-base').
        pretrained_path: Path to TAPT weights (optional).
        save_model_path: Path to save the best model checkpoint.
        val_preds_save_path: Path to save validation predictions (numpy).
        test_preds_save_path: Path to save test predictions (numpy).
        debug: Whether to run in debug mode.
    """
    print(f"\n=== Starting Supervised Training for {model_name} ===")

    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    print("Loading data...")
    train_df, val_df, test_df = load_raw_data(debug=debug)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create Datasets
    train_dataset = ToxicityDataset(
        train_df["comment_text"].tolist(),
        tokenizer,
        labels=train_df[Config.LABEL_COLS].values,
    )
    val_dataset = ToxicityDataset(
        val_df["comment_text"].tolist(),
        tokenizer,
        labels=val_df[Config.LABEL_COLS].values,
    )
    test_dataset = ToxicityDataset(
        test_df["comment_text"].tolist(),
        tokenizer,
        labels=None,  # Test set has no labels
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_PARAMS["batch_size"] * 2,  # Double batch size for eval
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TRAIN_PARAMS["batch_size"] * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CustomTransformer(
        model_name=model_name,
        num_classes=Config.NUM_LABELS,
        pretrained_path=pretrained_path,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_params = get_llrd_optimizer_params(
        model,
        lr_backbone=Config.TRAIN_PARAMS["lr_backbone"],
        lr_head=Config.TRAIN_PARAMS["lr_head"],
        weight_decay=Config.TRAIN_PARAMS["weight_decay"],
        llrd_decay=Config.TRAIN_PARAMS["llrd_decay"],
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    # Calculate total steps
    epochs = Config.TRAIN_PARAMS["epochs"]
    num_train_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        optimizer, num_train_steps, warmup_ratio=Config.TRAIN_PARAMS["warmup_ratio"]
    )

    # 5. Training Configuration
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()  # For Mixed Precision
    early_stopping = EarlyStopping(
        patience=Config.TRAIN_PARAMS["patience"], mode="max", save_path=save_model_path
    )

    # Intra-epoch validation logic
    val_check_interval = Config.TRAIN_PARAMS["val_check_interval"]
    eval_steps = int(len(train_loader) * val_check_interval)
    if eval_steps == 0:
        eval_steps = 1  # Safety check

    print(
        f"Training for {epochs} epochs. Steps per epoch: {len(train_loader)}. Validating every {eval_steps} steps."
    )

    # 6. Training Loop
    best_score = -np.inf

    for epoch in range(epochs):
        model.train()
        train_loss_meter = AverageMeter()
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step + 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Mixed Precision Forward
            with autocast(enabled=True):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            # Mixed Precision Backward
            scaler.scale(loss).backward()

            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.TRAIN_PARAMS["max_grad_norm"]
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_meter.update(loss.item(), input_ids.size(0))

            # Check for validation
            if (step + 1) % eval_steps == 0 or (step + 1) == len(train_loader):
                val_loss, val_auc = eval_fn(model, val_loader, device, loss_fn)

                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Train Loss: {train_loss_meter.avg:.5f} "
                    f"Val Loss: {val_loss:.5f} Val AUC: {val_auc:.8f}"
                )

                # Early Stopping Check
                early_stopping(val_auc, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

                # Revert to train mode
                model.train()

        if early_stopping.early_stop:
            break

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed:.1f}s")

    # 7. Load Best Model & Inference
    print(f"Loading best model from {save_model_path}...")
    # We need to handle DataParallel wrapping if it was used, but here we used single device logic
    # The EarlyStopping class handles saving state_dict
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    print("Generating predictions on Validation Set...")
    val_preds = inference_fn(model, val_loader, device)
    if val_preds_save_path:
        save_npy(val_preds, val_preds_save_path)
        print(f"Saved validation predictions to {val_preds_save_path}")

    print("Generating predictions on Test Set...")
    test_preds = inference_fn(model, test_loader, device)
    if test_preds_save_path:
        save_npy(test_preds, test_preds_save_path)
        print(f"Saved test predictions to {test_preds_save_path}")

    # Clear memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return val_preds, test_preds
