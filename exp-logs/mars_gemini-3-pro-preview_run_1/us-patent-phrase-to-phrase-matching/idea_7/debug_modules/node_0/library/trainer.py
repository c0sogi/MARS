import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, AdamW

from library.config import Config
from library.utils import get_logger, compute_score, seed_everything
from library.model import DebertaV3Regressor
from library.data import PhraseDataset

# Initialize logger
logger = get_logger(os.path.join(Config.working_dir, "train.log"))


def get_optimizer_params(model, encoder_lr, head_lr, weight_decay=0.01):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    # 1. Head Parameters
    head_params_decay = [
        p
        for n, p in model.named_parameters()
        if "head" in n and not any(nd in n for nd in no_decay)
    ]
    head_params_no_decay = [
        p
        for n, p in model.named_parameters()
        if "head" in n and any(nd in n for nd in no_decay)
    ]

    optimizer_parameters.append(
        {"params": head_params_decay, "lr": head_lr, "weight_decay": weight_decay}
    )
    optimizer_parameters.append(
        {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0}
    )

    # 2. Backbone Parameters (LLRD)
    # DeBERTa-v3-large typically has 24 layers.
    # Structure: backbone.encoder.layer.0 ... backbone.encoder.layer.23
    num_layers = model.config.num_hidden_layers

    # Iterate layers from top (closest to head) to bottom (closest to embeddings)
    for layer_i in range(num_layers - 1, -1, -1):
        layer_name = f"encoder.layer.{layer_i}."

        # Calculate decayed LR
        # Top layer gets encoder_lr, subsequent layers get decayed LR
        decay_rate = Config.llrd_decay ** (num_layers - 1 - layer_i)
        layer_lr = encoder_lr * decay_rate

        layer_params_decay = [
            p
            for n, p in model.named_parameters()
            if layer_name in n
            and "backbone" in n
            and not any(nd in n for nd in no_decay)
        ]
        layer_params_no_decay = [
            p
            for n, p in model.named_parameters()
            if layer_name in n and "backbone" in n and any(nd in n for nd in no_decay)
        ]

        if layer_params_decay:
            optimizer_parameters.append(
                {
                    "params": layer_params_decay,
                    "lr": layer_lr,
                    "weight_decay": weight_decay,
                }
            )
        if layer_params_no_decay:
            optimizer_parameters.append(
                {"params": layer_params_no_decay, "lr": layer_lr, "weight_decay": 0.0}
            )

    # Embeddings and initial projection layers get the lowest LR
    embedding_lr = encoder_lr * (Config.llrd_decay**num_layers)

    embedding_params_decay = [
        p
        for n, p in model.named_parameters()
        if "embeddings" in n and "backbone" in n and not any(nd in n for nd in no_decay)
    ]
    embedding_params_no_decay = [
        p
        for n, p in model.named_parameters()
        if "embeddings" in n and "backbone" in n and any(nd in n for nd in no_decay)
    ]

    # Catch-all for any other backbone parameters not caught by layer loop or embeddings check
    # (e.g. relative attention bias if separate)
    rest_params_decay = [
        p
        for n, p in model.named_parameters()
        if "backbone" in n
        and not any(x in n for x in ["head", "encoder.layer", "embeddings"])
        and not any(nd in n for nd in no_decay)
    ]
    rest_params_no_decay = [
        p
        for n, p in model.named_parameters()
        if "backbone" in n
        and not any(x in n for x in ["head", "encoder.layer", "embeddings"])
        and any(nd in n for nd in no_decay)
    ]

    if embedding_params_decay or rest_params_decay:
        optimizer_parameters.append(
            {
                "params": embedding_params_decay + rest_params_decay,
                "lr": embedding_lr,
                "weight_decay": weight_decay,
            }
        )
    if embedding_params_no_decay or rest_params_no_decay:
        optimizer_parameters.append(
            {
                "params": embedding_params_no_decay + rest_params_no_decay,
                "lr": embedding_lr,
                "weight_decay": 0.0,
            }
        )

    return optimizer_parameters


def train_fn(
    fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, scaler
):
    model.train()

    running_loss = 0.0
    count = 0

    start_time = time.time()

    # Gradient Accumulation setup
    accumulation_steps = Config.gradient_accumulation_steps

    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Mixed Precision Forward
        with torch.cuda.amp.autocast(enabled=Config.use_fp16):
            preds = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(preds, labels)

            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Step
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * accumulation_steps  # Scale back up for reporting
        count += batch_size

        if step % 100 == 0 and step > 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Fold {fold} | Epoch {epoch + 1} | Step {step}/{len(train_loader)} | "
                f"Loss: {running_loss / count:.4f} | Time: {elapsed:.0f}s"
            )

    return running_loss / count


def valid_fn(valid_loader, model, criterion, device):
    model.eval()

    preds = []
    labels = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            targets = batch["labels"].to(device)

            batch_size = targets.size(0)

            # Forward (Mixed Precision optional for inference)
            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            count += batch_size

            preds.extend(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    valid_loss = running_loss / count
    score = compute_score(labels, preds)

    return valid_loss, score, preds


def run_fold(fold, train_df, valid_df):
    logger.info(f"======== Running Fold {fold} ========")

    # 1. Prepare Data
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_dataset = PhraseDataset(
        train_df, tokenizer, max_length=Config.max_length, is_train=True
    )
    valid_dataset = PhraseDataset(
        valid_df, tokenizer, max_length=Config.max_length, is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 2. Model & Optimizer
    device = Config.device
    model = DebertaV3Regressor(Config.model_name, pretrained=True)
    model.to(device)

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.lr,
        head_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
    )

    optimizer = AdamW(optimizer_parameters)

    # Scheduler
    num_train_steps = int(
        len(train_loader) / Config.gradient_accumulation_steps * Config.epochs
    )
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        num_cycles=Config.num_cycles,
    )

    # Loss & Scaler
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)

    # 3. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.models_dir, f"model_fold_{fold}.pth")

    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_fn(
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            scaler,
        )
        valid_loss, valid_score, _ = valid_fn(valid_loader, model, criterion, device)

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Pearson: {valid_score} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if valid_score > best_score:
            best_score = valid_score
            logger.info(f"New Best Score! Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

    # Cleanup
    del (
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
    )
    torch.cuda.empty_cache()
    gc.collect()

    return best_score
