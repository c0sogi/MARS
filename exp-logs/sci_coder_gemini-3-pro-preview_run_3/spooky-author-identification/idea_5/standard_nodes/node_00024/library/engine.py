import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import MetricMonitor
from library.awp import AWP


def train_mlm(train_loader, model_name, output_dir, device, epochs=Config.MLM_EPOCHS):
    """
    Performs Masked Language Modeling (MLM) pre-training (Domain Adaptation).

    Args:
        train_loader: DataLoader containing the corpus.
        model_name: Name or path of the pre-trained model to start from.
        output_dir: Directory to save the adapted model.
        device: Torch device.
        epochs: Number of training epochs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model with MLM head
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    model.to(device)
    model.train()

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.MLM_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize GradScaler for Mixed Precision (Cite debug_lesson_7)
    scaler = GradScaler()

    accum_steps = Config.MLM_GRAD_ACCUMULATION_STEPS
    num_train_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    print(f"Starting MLM training for {model_name}...")

    for epoch in range(epochs):
        monitor = MetricMonitor()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mixed Precision Forward Pass
            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / accum_steps

            # Scaled Backward Pass
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                # Scaled Optimizer Step
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()

            monitor.update("loss", outputs.loss.item(), input_ids.size(0))

        print(f"MLM Epoch {epoch+1}/{epochs} | {monitor}")

    # Save the adapted model
    model.save_pretrained(output_dir)
    print(f"MLM Model saved to {output_dir}")


def train_fn(train_loader, model, optimizer, scheduler, epoch, device, use_awp=False):
    """
    Runs one epoch of supervised training, optionally with Adversarial Weight Perturbation (AWP).

    Args:
        train_loader: DataLoader for training data.
        model: The PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number (0-indexed).
        device: Torch device.
        use_awp: Boolean to enable AWP.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    monitor = MetricMonitor()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Initialize AWP only if enabled and past the start epoch
    awp = None
    if use_awp and epoch >= Config.AWP_START_EPOCH:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.AWP_LR,
            adv_eps=Config.AWP_EPS,
            start_epoch=Config.AWP_START_EPOCH,
            scaler=scaler,
        )

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # 1. Standard Forward Pass (Mixed Precision)
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        # 2. Standard Backward Pass (accumulate gradients)
        scaler.scale(loss).backward()

        # 3. Adversarial Training Step (AWP)
        if awp:
            # Save gradients and perturb weights (ascent)
            awp.attack_step()

            # Forward pass with perturbed weights
            with autocast():
                adv_logits = model(input_ids, attention_mask)
                adv_loss = criterion(adv_logits, labels)

            # Backward pass with perturbed weights (accumulate gradients)
            scaler.scale(adv_loss).backward()

            # Restore original weights
            awp.restore()

        # 4. Optimization Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        monitor.update("loss", loss.item(), input_ids.size(0))

    return monitor.metrics["loss"]["avg"]


def eval_fn(val_loader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader: DataLoader for validation data.
        model: The PyTorch model.
        device: Torch device.

    Returns:
        tuple: (average_loss, predictions)
    """
    model.eval()
    monitor = MetricMonitor()
    criterion = nn.CrossEntropyLoss()
    preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            monitor.update("loss", loss.item(), input_ids.size(0))

            # Calculate probabilities
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            preds.append(probs)

    predictions = np.concatenate(preds, axis=0)
    return monitor.metrics["loss"]["avg"], predictions


def inference_fn(dataloader, model, device):
    """
    Generates predictions for the test set.

    Args:
        dataloader: DataLoader for test data.
        model: The PyTorch model.
        device: Torch device.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            preds.append(probs)

    return np.concatenate(preds, axis=0)
