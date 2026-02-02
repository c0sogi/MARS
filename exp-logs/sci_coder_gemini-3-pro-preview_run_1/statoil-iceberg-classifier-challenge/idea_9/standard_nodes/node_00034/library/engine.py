import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_log_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss and Label Smoothing.

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
        epoch: Current epoch number (for printing)

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

        # Apply Label Smoothing (Cite solution_lesson_node_00033)
        # targets = targets * (1 - epsilon) + 0.5 * epsilon
        smooth_labels = (
            labels * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        )

        # Ensure labels are [B, 1]
        if smooth_labels.dim() == 1:
            smooth_labels = smooth_labels.view(-1, 1)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, smooth_labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {losses.avg}")
    return losses.avg


def validate_with_tta(model, loader, device):
    """
    Evaluates the model on the validation set using TTA (Original, HFlip, VFlip).

    Args:
        model: PyTorch model
        loader: DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        float: Log Loss score
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].cpu().numpy()

            # TTA View 1: Original
            logits_1 = model(images, angles)
            probs_1 = torch.sigmoid(logits_1)

            # TTA View 2: Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h, angles)
            probs_2 = torch.sigmoid(logits_2)

            # TTA View 3: Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v, angles)
            probs_3 = torch.sigmoid(logits_3)

            # Average Probabilities
            probs_avg = (probs_1 + probs_2 + probs_3) / 3.0

            all_preds.extend(probs_avg.cpu().numpy().flatten())
            all_targets.extend(labels)

    score = calculate_log_loss(all_targets, all_preds)
    print(f"Validation TTA Log Loss: {score}")
    return score


def predict_tta(model, loader, device):
    """
    Generates predictions for the test set using TTA (Original, HFlip, VFlip).

    Args:
        model: PyTorch model
        loader: DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        np.ndarray: Flattened array of predicted probabilities
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)

            # TTA View 1: Original
            logits_1 = model(images, angles)
            probs_1 = torch.sigmoid(logits_1)

            # TTA View 2: Horizontal Flip
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h, angles)
            probs_2 = torch.sigmoid(logits_2)

            # TTA View 3: Vertical Flip
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v, angles)
            probs_3 = torch.sigmoid(logits_3)

            # Average Probabilities
            probs_avg = (probs_1 + probs_2 + probs_3) / 3.0

            all_preds.extend(probs_avg.cpu().numpy().flatten())

    return np.array(all_preds)
