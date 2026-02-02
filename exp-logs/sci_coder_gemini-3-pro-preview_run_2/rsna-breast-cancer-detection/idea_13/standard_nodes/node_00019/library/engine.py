import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.config import Config


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pf1_score(y_true, y_pred):
    """
    Computes the Probabilistic F1 score (pF1).

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities [0, 1].

    Returns:
        float: The pF1 score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # pTP = Sum(Probability * Label)
    pTP = np.sum(y_pred * y_true)

    # pFP = Sum(Probability * (1 - Label))
    pFP = np.sum(y_pred * (1 - y_true))

    # Total Positives (TP + FN) in binary ground truth
    total_positives = np.sum(y_true)

    # Avoid division by zero
    epsilon = 1e-7

    # pPrecision = pTP / (pTP + pFP)
    p_precision = pTP / (pTP + pFP + epsilon)

    # pRecall = pTP / (TP + FN)
    p_recall = pTP / (total_positives + epsilon)

    # pF1 = 2 * (pPrec * pRec) / (pPrec + pRec)
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for step, (images, tabular, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast(enabled=True):
            logits = model(images, tabular)

            # FP32-Guarded Loss Calculation
            # We exit the autocast context and explicitly cast to float32
            # to prevent instability with high pos_weight
            with autocast(enabled=False):
                loss = criterion(logits.float(), labels.float())

        # Scale Loss and Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, labels in dataloader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            with autocast(enabled=True):
                logits = model(images, tabular)

                # FP32 Loss for consistency
                with autocast(enabled=False):
                    loss = criterion(logits.float(), labels.float())

            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).float().cpu().numpy()
            targets = labels.float().cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / len(dataloader)
    score = pf1_score(all_targets, all_preds)

    return avg_loss, score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_path,
):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # Initialize Loss with weighted positive class
    # pos_weight must be on the same device as the model/targets
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = GradScaler()

    best_score = -1.0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, criterion
        )

        # Validate
        val_loss, val_score = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score} -> {val_score}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training complete. Best Validation pF1: {best_score}")
