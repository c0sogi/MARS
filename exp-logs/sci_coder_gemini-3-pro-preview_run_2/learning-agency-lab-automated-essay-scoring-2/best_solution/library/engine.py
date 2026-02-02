import torch
import numpy as np
from library.config import Config


def train_fn(model, data_loader, optimizer, device, scheduler, criterion, scaler):
    """
    Performs a single epoch of training with Mixed Precision.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Device to run on.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        scaler: GradScaler for mixed precision.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(outputs, labels)

            # Gradient Accumulation
            if Config.GRAD_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRAD_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Record the true loss (scale back up)
        running_loss += loss.item() * Config.GRAD_ACCUMULATION_STEPS

    return running_loss / len(data_loader)


def eval_fn(model, data_loader, device, criterion=None):
    """
    Generates predictions and optionally computes loss.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for validation or test data.
        device: Device to run on.
        criterion: Loss function (optional).

    Returns:
        tuple: (predictions np.array, average_loss float or None)
    """
    model.eval()
    running_loss = 0.0
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask).squeeze(-1)

                # Calculate loss if labels are available and criterion is provided
                if criterion is not None and "labels" in batch:
                    labels = batch["labels"].to(device)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            preds.append(outputs.float().cpu().numpy())

    predictions = np.concatenate(preds)

    avg_loss = None
    if criterion is not None:
        avg_loss = running_loss / len(data_loader)

    return predictions, avg_loss
