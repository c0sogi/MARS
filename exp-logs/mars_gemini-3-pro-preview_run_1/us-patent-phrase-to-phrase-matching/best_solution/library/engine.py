import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_pearson


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Performs one epoch of training using FP16 mixed precision and gradient accumulation.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training data.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to run on (cuda/cpu).
        scaler: torch.cuda.amp.GradScaler for mixed precision.
        epoch: Current epoch number (integer).

    Returns:
        tuple: (average_loss, pearson_score)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    accumulation_steps = Config.gradient_accumulation_steps
    num_batches = len(dataloader)

    # Ensure gradients are zero at the start of the epoch
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            # Normalize loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Track metrics (use item() * batch_size for accurate average)
        # Note: outputs.loss is the mean loss for the batch
        batch_loss = outputs.loss.item()
        running_loss += batch_loss * input_ids.size(0)

        logits = outputs.logits.squeeze(-1)
        all_preds.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # Optimization Step (Gradient Accumulation)
        if (step + 1) % accumulation_steps == 0 or (step + 1) == num_batches:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Reset gradients and step scheduler
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_pearson = compute_pearson(all_preds, all_labels)

    return epoch_loss, epoch_pearson


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the validation data.
        device: The device to run on.

    Returns:
        tuple: (average_loss, pearson_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits.squeeze(-1)

            running_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_pearson = compute_pearson(all_preds, all_labels)

    return epoch_loss, epoch_pearson


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the test data.
        device: The device to run on.

    Returns:
        np.ndarray: Array of predicted scores.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
            )

            logits = outputs.logits.squeeze(-1)
            all_preds.extend(logits.cpu().numpy())

    return np.array(all_preds)
