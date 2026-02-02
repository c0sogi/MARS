import torch
import numpy as np
from library.config import Config


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler=None,
    max_grad_norm=Config.max_grad_norm,
):
    """
    Handles the training loop for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        criterion: The loss function.
        device: The device to run on.
        scaler: The GradScaler for AMP.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        float: The average training loss.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Handles the validation loop for one epoch.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, predictions, true_labels)
    """
    model.eval()
    total_loss = 0.0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply sigmoid for predictions
            probs = torch.sigmoid(logits)
            preds_list.append(probs.float().cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    final_preds = np.concatenate(preds_list, axis=0)
    final_labels = np.concatenate(labels_list, axis=0)

    return avg_loss, final_preds, final_labels
