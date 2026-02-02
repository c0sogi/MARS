import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from library.config import Config
from library.utils import seed_everything, compute_pearson_score
from library.data import prepare_loaders
from library.model import CustomDeberta
from library.awp import AWP


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    scaler=None,
):
    model.train()

    losses = []
    start = time.time()

    # Global step tracking if needed, but we rely on loader length
    num_steps = len(train_loader)

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # --- Forward Pass (Clean) ---
        with autocast(enabled=True):
            y_preds = model(input_ids, attention_mask)
            loss = criterion(y_preds, labels)

        # --- Backward Pass (Clean) ---
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # --- Adversarial Weight Perturbation (AWP) ---
        if awp is not None and epoch >= Config.awp_start_epoch:
            # 1. Perturb weights based on current gradients
            awp.attack_step()

            # 2. Forward pass with perturbed weights
            with autocast(enabled=True):
                y_preds_adv = model(input_ids, attention_mask)
                loss_adv = criterion(y_preds_adv, labels)

            # 3. Backward pass with perturbed weights
            # We accumulate gradients (clean + adv)
            if scaler:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            # 4. Restore original weights
            awp._restore()

        # --- Optimizer Step ---
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss


def valid_fn(valid_loader, model, criterion, device):
    model.eval()

    losses = []
    preds = []
    labels_list = []

    # Class values for Expected Value calculation: 0.0, 0.25, 0.5, 0.75, 1.0
    class_values = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.00], device=device)

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=True):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            losses.append(loss.item())

            # Calculate Expected Value
            # Softmax -> Probabilities -> Weighted Sum
            probs = torch.softmax(logits, dim=1)
            batch_preds = torch.sum(probs * class_values, dim=1)

            preds.append(batch_preds.cpu().numpy())

            # Convert labels back to scores for metric calculation
            # Labels are indices 0-4. Map back to 0.0-1.0
            # Or use the original scores if available, but dataset returns class indices.
            # We can map indices back: index / 4.0
            batch_labels = labels.cpu().numpy() / 4.0
            labels_list.append(batch_labels)

    avg_loss = np.mean(losses)
    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(labels_list)

    return avg_loss, predictions, ground_truth


def run_training():
    seed_everything(Config.seed)

    print(f"Initializing training on device: {Config.device}")

    # --- Data Preparation ---
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_loader, val_loader, test_loader = prepare_loaders(
        tokenizer, load_cached_data=True
    )

    # --- Model Initialization ---
    model = CustomDeberta(Config.model_name, pretrained=True)
    model.to(Config.device)

    # --- Optimizer & Scheduler ---
    # Separate weight decay for bias/LayerNorm
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_parameters,
        lr=Config.learning_rate,
        eps=Config.eps,
        betas=Config.betas,
    )

    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # --- Loss Function ---
    # Using CrossEntropyLoss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # --- AWP Initialization ---
    scaler = GradScaler()
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
            scaler=scaler,
        )

    # --- Training Loop ---
    best_score = -1.0
    best_loss = np.inf

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            Config.device,
            awp,
            scaler,
        )

        # Validate
        val_loss, val_preds, val_labels = valid_fn(
            val_loader, model, criterion, Config.device
        )

        # Metric
        val_score = compute_pearson_score(val_labels, val_preds)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Pearson: {val_score:.16f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            print(f"New Best Score! Saving model...")
            torch.save(
                model.state_dict(), os.path.join(Config.output_dir, f"model_fold_0.bin")
            )

    print(f"Training Complete. Best Pearson Score: {best_score:.16f}")

    # Free memory
    del model, optimizer, scheduler, awp, scaler
    torch.cuda.empty_cache()
    gc.collect()
