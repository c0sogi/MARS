import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from library.config import Config
from library.utils import AverageMeter, get_score
from library.data import get_tokenizer


def train_mlm(loader, device, epochs=Config.mlm_epochs):
    """
    Performs Domain-Adaptive Pre-training (MLM) on the provided loader.
    Saves the fine-tuned backbone to Config.mlm_model_dir.
    """
    print(f"Initializing MLM training for {epochs} epochs...")

    # Initialize Model and Tokenizer
    # We use the base model configuration
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(device)
    model.train()

    tokenizer = get_tokenizer()

    # Optimizer for MLM
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.mlm_lr, weight_decay=Config.mlm_weight_decay
    )
    scaler = GradScaler()

    # Data Collator for masking
    # We use the standard 15% masking probability
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    accumulation_steps = getattr(Config, "mlm_grad_accum", 1)

    for epoch in range(epochs):
        losses = AverageMeter()
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            # The loader yields batches of input_ids and attention_mask (tensors)
            # We need to convert them to a list of dicts for the DataCollator
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_size = input_ids.size(0)

            # Prepare inputs for collator
            examples = [
                {"input_ids": input_ids[i], "attention_mask": attention_mask[i]}
                for i in range(batch_size)
            ]

            # Apply masking
            # collator returns dict with keys: input_ids, attention_mask, labels
            masked_batch = collator(examples)

            # Move to device
            b_input_ids = masked_batch["input_ids"].to(device)
            b_attention_mask = masked_batch["attention_mask"].to(device)
            b_labels = masked_batch["labels"].to(device)

            # Forward
            with autocast():
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels,
                )
                loss = outputs.loss / accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses.update(loss.item() * accumulation_steps, batch_size)

        print(f"MLM Epoch {epoch + 1}/{epochs} | Loss: {losses.avg:.6f}")

    # Save the domain-adapted model
    print(f"Saving MLM backbone to {Config.mlm_model_dir}...")
    os.makedirs(Config.mlm_model_dir, exist_ok=True)
    model.save_pretrained(Config.mlm_model_dir)
    tokenizer.save_pretrained(Config.mlm_model_dir)
    print("MLM training complete.")


def train_fn(loader, model, optimizer, scheduler, device, epoch, awp=None):
    """
    Training loop for one epoch of supervised classification.
    Integrates Adversarial Weight Perturbation (AWP).
    """
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        # 1. Standard Forward Pass
        with autocast():
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

        # 2. Standard Backward Pass (Compute Gradients)
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # 3. Adversarial Weight Perturbation
        if awp is not None and epoch >= awp.start_epoch:
            # Save weights and apply perturbation based on current gradients
            awp.attack_step()

            # Forward pass with perturbed weights
            with autocast():
                outputs_adv = model(input_ids, attention_mask, labels)
                loss_adv = outputs_adv["loss"]

            # Backward pass with perturbed weights
            # We zero grad to use only the adversarial gradient for the update
            # This minimizes the loss at the worst-case perturbation
            optimizer.zero_grad()
            scaler.scale(loss_adv).backward()

            # Restore original weights (gradients remain accumulated)
            awp._restore()

        # 4. Optimization Step
        # Clip gradients to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def valid_fn(loader, model, device):
    """
    Validation loop. Calculates Loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            losses.update(loss.item(), batch_size)

            # Store predictions and targets
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(labels.float().cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate Metric
    score = get_score(targets, preds)

    return losses.avg, score


def inference_fn(loader, model, device):
    """
    Inference loop for generating predictions on the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, labels=None)
            logits = outputs["logits"]

            preds.append(torch.sigmoid(logits).float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds
