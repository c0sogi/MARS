import torch
import torch.nn as nn
import numpy as np
import gc
from transformers import AutoModelForMaskedLM, AdamW

from library.config import Config
from library.utils import AverageMeter, get_score
from library.awp import AWP


def train_mlm(train_loader, device):
    """
    Performs Domain-Adaptive Pre-training (DAPT) using Masked Language Modeling (MLM).
    Trains an AutoModelForMaskedLM on the combined corpus and saves the backbone.
    """
    print(f"Starting Domain-Adaptive Pre-training for {Config.mlm_epochs} epochs...")

    # Initialize model for MLM from base configuration
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(device)
    model.train()

    # Optimizer for Pre-training
    optimizer = AdamW(
        model.parameters(), lr=Config.mlm_lr, weight_decay=Config.weight_decay
    )

    for epoch in range(Config.mlm_epochs):
        losses = AverageMeter()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Update metrics
            losses.update(loss.item(), input_ids.size(0))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"MLM Epoch {epoch+1}/{Config.mlm_epochs} - Loss: {losses.avg}")

    # Save the domain-adapted model
    print(f"Saving MLM backbone to {Config.mlm_model_dir}...")
    model.save_pretrained(Config.mlm_model_dir)

    # Cleanup to free memory for the main training stage
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()


def train_fn(
    train_loader, model, criterion, optimizer, scheduler, device, epoch, awp=None
):
    """
    Executes one training epoch for Supervised Fine-Tuning.
    Implements Adversarial Weight Perturbation (AWP) if enabled.
    """
    model.train()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        # --- 1. Standard Forward & Backward Pass ---
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        losses.update(loss.item(), batch_size)
        loss.backward()

        # --- 2. Adversarial Weight Perturbation (AWP) ---
        if Config.use_awp and awp is not None and epoch >= Config.awp_start_epoch:
            # Perturb weights to maximize loss (ascent)
            awp._attack()

            # Forward pass with perturbed weights
            adv_outputs = model(input_ids, attention_mask, labels)
            adv_loss = adv_outputs["loss"]

            # Calculate gradients at the perturbed point
            # We zero out the original gradients to update based on the worst-case scenario
            optimizer.zero_grad()
            adv_loss.backward()

            # Restore original weights (but keep the adversarial gradients)
            awp._restore()

        # --- 3. Optimization Step ---
        nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return losses.avg


def valid_fn(val_loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            losses.update(loss.item(), batch_size)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            preds.append(probs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate metric
    score = get_score(targets, preds)

    return losses.avg, score


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, labels=None)
            logits = outputs["logits"]

            probs = torch.sigmoid(logits)
            preds.append(probs.detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds
