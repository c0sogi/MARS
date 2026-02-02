import os
import gc
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    Logger,
    save_checkpoint,
    load_checkpoint,
    get_score,
)
from library.data import load_and_process_data, InsultDataset
from library.model import InsultModel
from library.awp import AWP


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures Layer-wise Learning Rate Decay (LLRD) for the optimizer.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Base LLRD settings
    # DeBERTa-v3-large has 24 layers.
    # Structure: backbone.embeddings, backbone.encoder.layer.{0..23}, fc (head)
    num_layers = 24
    decay_rate = Config.llrd_decay

    optimizer_parameters = []

    for name, p in param_optimizer:
        if not p.requires_grad:
            continue

        # Determine layer depth
        if "embeddings" in name:
            depth = 0
        elif "encoder.layer" in name:
            # Extract layer index
            # Example: backbone.encoder.layer.12.output...
            parts = name.split(".")
            found = False
            for part in parts:
                if part.isdigit():
                    depth = int(part) + 1
                    found = True
                    break
            if not found:
                depth = 0
        elif "fc" in name or "pooler" in name:
            depth = num_layers + 1
        else:
            # Other parameters (e.g. final layer norm in backbone)
            depth = num_layers

        # Calculate LR for this parameter
        # Head gets decoder_lr
        # Layers get decayed lr
        if depth == num_layers + 1:
            lr = decoder_lr
        else:
            lr = encoder_lr * (decay_rate ** (num_layers + 1 - depth))

        # Determine weight decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        optimizer_parameters.append({"params": [p], "lr": lr, "weight_decay": wd})

    return optimizer_parameters


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, awp, epoch):
    model.train()
    loss_meter = AverageMeter()

    # Loss function: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        # 1. Forward pass (Clean)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        # 2. Backward pass (Clean)
        scaler.scale(loss).backward()

        # 3. Adversarial Weight Perturbation (AWP)
        # Apply AWP only after specified epoch
        if Config.use_awp and epoch >= Config.awp_start_epoch:
            # Save weights and perturb based on gradients
            awp.attack()

            # Forward pass (Adversarial)
            with torch.cuda.amp.autocast(enabled=True):
                outputs_adv = model(input_ids, attention_mask)
                loss_adv = criterion(outputs_adv, labels)

            # Backward pass (Adversarial)
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # 4. Optimizer Step
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), batch_size)

    return loss_meter.avg


def valid_fn(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs).reshape(-1)

        preds.append(probs.cpu().numpy())
        targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return preds, targets


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).reshape(-1)

        preds.append(probs.cpu().numpy())

    return np.concatenate(preds)


def run_fold(fold, df_train, df_val, tokenizer, device, logger):
    logger.log(f"=== Running Fold {fold} ===")

    # Datasets
    train_dataset = InsultDataset(df_train, tokenizer, Config.max_len)
    val_dataset = InsultDataset(df_val, tokenizer, Config.max_len)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model Initialization
    model = InsultModel(pretrained=True)
    model.to(device)

    # Load TAPT weights if available and requested
    if Config.use_tapt:
        tapt_path = os.path.join(Config.tapt_output_dir, "tapt_model.pth")
        if os.path.exists(tapt_path):
            logger.log(f"Loading TAPT weights from {tapt_path}")
            model.load_tapt_weights(tapt_path)
        else:
            logger.log(
                f"TAPT weights not found at {tapt_path}. Proceeding with base weights."
            )

    # Optimizer with LLRD
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.eps
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Scaler & AWP
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    # Training Loop
    best_score = -np.inf
    best_model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")

    for epoch in range(1, Config.epochs + 1):
        start_time = time.time()

        avg_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, scaler, awp, epoch
        )
        preds, targets = valid_fn(model, val_loader, device)
        score = get_score(targets, preds)

        elapsed = time.time() - start_time
        logger.log(
            f"Epoch {epoch} - avg_loss: {avg_loss:.4f} - val_auc: {score:.8f} - time: {elapsed:.0f}s"
        )

        if score > best_score:
            best_score = score
            logger.log(f"Epoch {epoch} - Save Best Score: {best_score:.8f}")
            torch.save(model.state_dict(), best_model_path)

    # Load best model for OOF predictions
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    oof_preds, oof_targets = valid_fn(model, val_loader, device)

    # Cleanup
    del model, optimizer, scheduler, scaler, awp, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return oof_preds, oof_targets, best_score


def run_training():
    seed_everything(Config.seed)
    Config.setup()

    logger = Logger(os.path.join(Config.output_dir, "train_log.txt"))
    logger.log("Starting Training Pipeline...")

    # Load Data
    df_train_meta, df_val_meta, df_test = load_and_process_data(load_cached_data=True)

    # Combine train and val for Cross-Validation
    df_full = pd.concat([df_train_meta, df_val_meta]).reset_index(drop=True)

    # Debug Mode
    if Config.debug:
        logger.log(f"DEBUG MODE: Subsetting data to {Config.debug_subset_size} rows.")
        df_full = df_full.head(Config.debug_subset_size)
        df_test = df_test.head(Config.debug_subset_size)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    oof_preds_full = np.zeros(len(df_full))
    test_preds_accum = np.zeros(len(df_test))

    # Iterate Folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full[Config.target_col])
    ):
        df_train_fold = df_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_full.iloc[val_idx].reset_index(drop=True)

        # Run Fold
        oof_preds, oof_targets, best_score = run_fold(
            fold, df_train_fold, df_val_fold, tokenizer, Config.device, logger
        )

        # Store OOF
        oof_preds_full[val_idx] = oof_preds

        # Inference on Test Set
        logger.log(f"Generating test predictions for Fold {fold}...")

        # Load best model
        model = InsultModel(pretrained=False)
        model.to(Config.device)
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=Config.device))

        test_dataset = InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        fold_test_preds = inference_fn(model, test_loader, Config.device)
        test_preds_accum += fold_test_preds / Config.n_folds

        del model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # Overall CV Score
    overall_score = get_score(df_full[Config.target_col].values, oof_preds_full)
    logger.log(f"Overall CV AUC: {overall_score:.8f}")

    # Save Submission
    submission = pd.DataFrame()
    # Assuming sample_submission format or just ID/Pred.
    # The task description says "Your predictions should be a number in the range [0,1]."
    # And "See 'sample_submissions_null.csv' for the correct format."
    # Usually we can just output the probabilities.
    # We will create a submission with the same index/structure as test.csv or sample_submission.

    # Let's check sample submission structure from description
    # It has columns: Insult, Date, Comment.
    # Usually we need to fill the 'Insult' column.

    submission = df_test.copy()
    submission["Insult"] = test_preds_accum

    # Ensure correct columns are saved (usually ID and Target, but here we might just save the full CSV or specific format)
    # The prompt says "Your predictions should be a number in the range [0,1]."
    # We will save the full dataframe with updated 'Insult' column to be safe, or just the required columns.
    # Given the input format, let's stick to the structure of sample_submission_null.csv

    # If sample_submission_null.csv has 'Insult', 'Date', 'Comment', we should probably keep them.
    # However, standard Kaggle submissions are usually ID, Prediction.
    # Here, no ID column exists.
    # We will save the 'Insult' column as the prediction.

    # To be safe and compliant with "submission.csv" expectation:
    # We will save the dataframe with 'Insult' updated.

    submission.to_csv(Config.submission_path, index=False)
    logger.log(f"Submission saved to {Config.submission_path}")
