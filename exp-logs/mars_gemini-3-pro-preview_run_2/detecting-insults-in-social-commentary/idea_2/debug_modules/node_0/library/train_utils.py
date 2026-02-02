import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to use for training.
        loss_fn (torch.nn.Module): The loss function.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # BCEWithLogitsLoss expects labels to be the same shape as logits (batch_size, 1)
        labels = labels.unsqueeze(1).float()

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to use for evaluation.
        loss_fn (torch.nn.Module): The loss function.

    Returns:
        tuple: A tuple containing (average_loss, auc_score).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            labels = labels.unsqueeze(1).float()

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            running_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / len(dataloader)

    # Calculate AUC
    # Handle edge case where only one class is present in the batch/subset
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        dataloader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to use.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_preds)
