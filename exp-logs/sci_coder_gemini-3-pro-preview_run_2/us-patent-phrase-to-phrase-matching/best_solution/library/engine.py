import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, compute_score


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    epoch,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
):
    """
    Trains the model for one epoch with gradient accumulation and clipping.
    """
    model.train()

    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["label"].to(device)

        batch_size = input_ids.size(0)

        # Forward pass
        outputs = model(input_ids, attention_mask, structural_features)

        loss = criterion(outputs, labels)

        losses.update(loss.item(), batch_size)

        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update weights after accumulation steps
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def validate_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set using Pearson correlation.
    Computes the Expected Value of the class probabilities to get a continuous score.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The device to run evaluation on.

    Returns:
        tuple: (average_loss, pearson_score)
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    preds_all = []
    labels_all = []

    # The scores corresponding to classes 0, 1, 2, 3, 4
    score_values = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], device=device)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structural_features = batch["structural_features"].to(device)
            labels = batch["label"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask, structural_features)

            loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)

            # Calculate Expected Value: Sum(P(c) * Score(c))
            probs = torch.softmax(outputs, dim=1)
            expected_scores = torch.sum(probs * score_values, dim=1)

            preds_all.append(expected_scores.cpu().numpy())

            # Convert integer class labels back to float scores for metric calculation
            # Class 0->0.0, 1->0.25, etc.
            true_scores = labels.float() * 0.25
            labels_all.append(true_scores.cpu().numpy())

    preds_flat = np.concatenate(preds_all)
    labels_flat = np.concatenate(labels_all)

    pearson_score = compute_score(labels_flat, preds_flat)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Pearson: {pearson_score}")

    return losses.avg, pearson_score


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset (e.g., test set).

    Args:
        model: The PyTorch model.
        dataloader: The dataloader (labels not required).
        device: The device.

    Returns:
        np.ndarray: Array of predicted scores.
    """
    model.eval()

    preds_all = []
    score_values = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], device=device)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structural_features = batch["structural_features"].to(device)

            outputs = model(input_ids, attention_mask, structural_features)

            probs = torch.softmax(outputs, dim=1)
            expected_scores = torch.sum(probs * score_values, dim=1)

            preds_all.append(expected_scores.cpu().numpy())

    return np.concatenate(preds_all)
