import time
import numpy as np
import torch
import torch.nn as nn
from library.utils import AverageMeter, get_score, time_since
from library.model import AWP


def train_mlm(model, dataloader, optimizer, scheduler, device, epoch, config):
    """
    Trains a Masked Language Model (MLM) for Domain Adaptation.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    print(f"\n[DAPT] Epoch {epoch + 1}/{config.dapt_epochs} Training...")

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Forward pass (HF models return loss when labels are provided)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        losses.update(loss.item(), batch_size)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        if (step + 1) % 100 == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch {epoch + 1} Step {step + 1}/{len(dataloader)} "
                f"Loss: {losses.avg:.4f} "
                f"Time: {time_since(start, (step + 1) / len(dataloader))}"
            )

    return losses.avg


def train_fn(model, dataloader, optimizer, scheduler, device, epoch, config):
    """
    Trains the classification model (Teacher or Student) with optional AWP.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    # Initialize AWP if configured
    awp = None
    if config.use_awp and epoch >= config.awp_start_epoch:
        awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    criterion = nn.BCEWithLogitsLoss()

    print(f"\n[Train] Epoch {epoch + 1} Training...")

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # --- Standard Step ---
        y_preds = model(input_ids, attention_mask)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        loss.backward()

        # --- AWP Step ---
        if awp is not None:
            # Perturb weights
            awp.attack_step()

            # Forward pass with perturbed weights
            y_preds_adv = model(input_ids, attention_mask)
            loss_adv = criterion(y_preds_adv, labels)

            # Use the adversarial gradients for the update
            optimizer.zero_grad()
            loss_adv.backward()

            # Restore original weights
            awp.restore()

        # --- Optimization ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % 100 == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch {epoch + 1} Step {step + 1}/{len(dataloader)} "
                f"Loss: {losses.avg:.4f} "
                f"Time: {time_since(start, (step + 1) / len(dataloader))}"
            )

    return losses.avg


def valid_fn(model, dataloader, device, config):
    """
    Evaluates the model on the validation set.
    Returns average loss and predictions.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    start = time.time()

    criterion = nn.BCEWithLogitsLoss()

    print("\n[Valid] Evaluating...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            y_preds = model(input_ids, attention_mask)
            loss = criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)

            # Apply sigmoid to convert logits to probabilities
            preds.append(y_preds.sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)

    print(f"[Valid] Result - Avg Loss: {losses.avg:.6f}")

    return losses.avg, predictions


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    start = time.time()

    print("\n[Inference] Generating predictions...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_preds = model(input_ids, attention_mask)

            # Apply sigmoid to convert logits to probabilities
            preds.append(y_preds.sigmoid().to("cpu").numpy())

            if (step + 1) % 100 == 0:
                print(f"Inference Step {step + 1}/{len(dataloader)}")

    predictions = np.concatenate(preds)
    return predictions
