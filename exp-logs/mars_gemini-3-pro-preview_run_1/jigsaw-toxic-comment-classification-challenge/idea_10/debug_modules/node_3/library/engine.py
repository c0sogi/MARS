import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast  # Cite {debug_lesson_10}
from transformers import (
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from library.config import Config
from library.utils import get_score, AWP
from library.data import get_mlm_loader


def run_mlm():
    """
    Executes Domain-Adaptive Pre-training (DAPT) using Masked Language Modeling.
    Saves the fine-tuned backbone to Config.dapt_model_path.
    """
    if not Config.train_dapt:
        print("Skipping DAPT as per configuration.")
        return

    print(f"Starting Domain-Adaptive Pre-training (MLM) on {Config.device}...")

    # Initialize directory
    os.makedirs(Config.dapt_model_path, exist_ok=True)

    # Load Model for MLM
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(Config.device)
    model.train()

    # Load Data
    # We pass the tokenizer from the model to ensure compatibility
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_loader = get_mlm_loader(tokenizer, load_cached_data=True)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.dapt_lr, weight_decay=Config.dapt_weight_decay
    )

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.dapt_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * 0.1),
        num_training_steps=num_train_steps,
    )

    scaler = GradScaler()  # Cite {debug_lesson_10}

    # Training Loop
    for epoch in range(Config.dapt_epochs):
        start_time = time.time()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            optimizer.zero_grad()

            with autocast():  # Cite {debug_lesson_10}
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % Config.print_freq == 0:
                print(
                    f"Epoch [{epoch+1}/{Config.dapt_epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} DAPT Complete. Avg Loss: {avg_loss:.4f} Time: {elapsed:.0f}s"
        )

    # Save the domain-adapted model
    print(f"Saving DAPT model to {Config.dapt_model_path}...")
    model.save_pretrained(Config.dapt_model_path)
    tokenizer.save_pretrained(Config.dapt_model_path)
    print("DAPT Complete.")


def train_fn(
    model, data_loader, optimizer, scheduler, device, epoch, awp=None, scaler=None
):
    """
    Performs one epoch of supervised training.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    start_time = time.time()

    # Enable AWP if configured and epoch threshold is met
    use_awp = False
    if Config.use_awp and awp is not None and epoch >= Config.awp_start_epoch:
        use_awp = True
        print(f"  [Epoch {epoch+1}] AWP Enabled.")

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():  # Cite {debug_lesson_10}
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adversarial Weight Perturbation
        if use_awp:
            if scaler:
                scaler.unscale_(optimizer)

            # Save weights and apply perturbation based on gradients
            awp.attack()

            # Forward pass with perturbed weights
            with autocast():
                logits_adv = model(input_ids, attention_mask)
                loss_adv = criterion(logits_adv, labels)

            # Backward pass for adversarial loss
            if scaler:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            # Restore original weights
            awp._restore()

        # Gradient Clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % Config.print_freq == 0:
            print(f"  Step [{step+1}/{len(data_loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    elapsed = time.time() - start_time

    return avg_loss


def valid_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    preds = []
    targets = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():  # Cite {debug_lesson_10}
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            preds.append(probs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    score = get_score(targets, preds)
    elapsed = time.time() - start_time

    return avg_loss, score
