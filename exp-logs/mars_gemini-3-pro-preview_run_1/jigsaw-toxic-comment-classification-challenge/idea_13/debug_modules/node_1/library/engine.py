import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, get_score


def train_mlm(model, train_loader, optimizer, scheduler, device, epoch, logger=None):
    """
    Trains the model for Masked Language Modeling (DAPT).
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # Forward pass (HuggingFace models compute MLM loss automatically if labels are provided)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        losses.update(loss.item(), batch_size)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            if logger:
                logger.info(
                    f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed {time.time() - start:.1f}s "
                    f"Loss: {losses.val:.4f} Avg: {losses.avg:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

    return losses.avg


def train_fn(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    logger=None,
):
    """
    Trains the model for one epoch using Deep Supervision and optionally AWP.
    Used for both Supervised Teacher training and Student Distillation.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # --- Forward Pass ---
        outputs = model(input_ids, attention_mask)
        main_logits = outputs["main_logits"]
        aux_logits = outputs["aux_logits"]

        # --- Loss Calculation ---
        loss_main = criterion(main_logits, labels)

        # Deep Supervision Logic
        if Config.use_deep_supervision:
            loss_aux = criterion(aux_logits, labels)
            loss = loss_main + Config.aux_loss_weight * loss_aux
        else:
            loss = loss_main

        losses.update(loss.item(), batch_size)

        # --- Backward Pass ---
        loss.backward()

        # --- AWP Attack ---
        # AWP is applied after the first backward pass to perturb weights based on gradients
        if awp is not None and (epoch + 1) >= Config.awp_start_epoch:
            awp._save()
            awp._attack_step()

            # Re-forward with perturbed weights
            outputs_adv = model(input_ids, attention_mask)
            main_logits_adv = outputs_adv["main_logits"]
            aux_logits_adv = outputs_adv["aux_logits"]

            loss_main_adv = criterion(main_logits_adv, labels)

            if Config.use_deep_supervision:
                loss_aux_adv = criterion(aux_logits_adv, labels)
                loss_adv = loss_main_adv + Config.aux_loss_weight * loss_aux_adv
            else:
                loss_adv = loss_main_adv

            # Accumulate gradients from adversarial pass
            loss_adv.backward()

            # Restore original weights
            awp.restore()

        # --- Optimizer Step ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            if logger:
                logger.info(
                    f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed {time.time() - start:.1f}s "
                    f"Loss: {losses.val:.4f} Avg: {losses.avg:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

    return losses.avg


def valid_fn(val_loader, model, criterion, device, logger=None):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    for step, data in enumerate(val_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            main_logits = outputs["main_logits"]
            # Validation metric is based only on the main head

            loss = criterion(main_logits, labels)

        losses.update(loss.item(), batch_size)

        # Apply sigmoid to get probabilities
        preds.append(torch.sigmoid(main_logits).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

        if step % Config.print_freq == 0 or step == (len(val_loader) - 1):
            if logger:
                logger.info(
                    f"EVAL: [{step}/{len(val_loader)}] "
                    f"Elapsed {time.time() - start:.1f}s "
                    f"Loss: {losses.val:.4f} Avg: {losses.avg:.4f} "
                )

    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate metric
    score = get_score(targets, predictions)

    if logger:
        logger.info(f"Validation Loss: {losses.avg}")
        logger.info(f"Validation Score (AUC): {score}")

    return losses.avg, predictions, targets


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    for step, data in enumerate(test_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            main_logits = outputs["main_logits"]

        preds.append(torch.sigmoid(main_logits).detach().cpu().numpy())

    predictions = np.concatenate(preds)
    return predictions
