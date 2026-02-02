import time
import torch
import torch.nn as nn
import numpy as np
from library.config import CFG
from library.utils import AverageMeter, get_score, AWP


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """
    Executes one training epoch.
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = AverageMeter()

    # Initialize AWP (Adversarial Weight Perturbation)
    awp = AWP(
        model,
        optimizer,
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
        scaler=scaler,
    )

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["label"].to(device)

        # Convert float labels (0.0, 0.25, ...) to class indices (0, 1, ...)
        target_indices = (labels * 4).round().long()
        batch_size = labels.size(0)

        # Standard Forward Pass
        with torch.amp.autocast("cuda", enabled=True):
            y_preds = model(input_ids, attention_mask, structural_features)
            loss = criterion(y_preds, target_indices)

        # Scale loss for gradient accumulation
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        losses.update(loss.item() * CFG.gradient_accumulation_steps, batch_size)

        # Backward Pass
        scaler.scale(loss).backward()

        # AWP Attack Step
        # Only execute if AWP is enabled and we are past the start epoch
        if CFG.use_awp and epoch >= CFG.awp_start_epoch:
            awp.attack()
            with torch.amp.autocast("cuda", enabled=True):
                y_preds_adv = model(input_ids, attention_mask, structural_features)
                loss_adv = criterion(y_preds_adv, target_indices)
                if CFG.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / CFG.gradient_accumulation_steps
            scaler.scale(loss_adv).backward()
            awp._restore()

        # Optimizer Step (with Gradient Accumulation)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    return losses.avg


def valid_fn(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns loss, pearson score, and raw probability predictions.
    """
    model.eval()
    losses = AverageMeter()
    preds_probs = []
    true_labels = []

    # Values corresponding to the 5 classes for scalar conversion
    score_vals_np = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["label"].to(device)

        target_indices = (labels * 4).round().long()
        batch_size = labels.size(0)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                y_preds = model(input_ids, attention_mask, structural_features)
                loss = criterion(y_preds, target_indices)

        losses.update(loss.item(), batch_size)

        # Convert logits to probabilities
        probs = torch.softmax(y_preds, dim=1)

        preds_probs.append(probs.to("cpu").numpy())
        true_labels.append(labels.to("cpu").numpy())

    preds_probs = np.concatenate(preds_probs)
    true_labels = np.concatenate(true_labels)

    # Calculate scalar predictions via expectation: sum(prob * value)
    preds_scalar = np.sum(preds_probs * score_vals_np, axis=1)

    # Calculate Pearson Correlation
    score = get_score(true_labels, preds_scalar)

    return losses.avg, score, preds_probs


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    Returns raw probability distributions.
    """
    model.eval()
    preds_probs = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                y_preds = model(input_ids, attention_mask, structural_features)

        probs = torch.softmax(y_preds, dim=1)
        preds_probs.append(probs.to("cpu").numpy())

    preds_probs = np.concatenate(preds_probs)
    return preds_probs
