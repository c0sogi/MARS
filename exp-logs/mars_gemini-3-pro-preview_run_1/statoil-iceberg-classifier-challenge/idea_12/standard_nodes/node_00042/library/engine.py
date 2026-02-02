import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, calculate_log_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Executes one epoch of training.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer (e.g., AdamW).
        device (str): Device to run on ('cuda' or 'cpu').
        epoch (int): Current epoch number (for logging).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # Binary Cross Entropy with Logits for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, angles, targets) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        # Targets need to be (B, 1) float to match logits
        targets = targets.to(device).float().unsqueeze(1)

        # Apply Label Smoothing (Cite solution_lesson_node_00005)
        smooth_targets = (
            targets * (1 - 2 * Config.LABEL_SMOOTHING) + Config.LABEL_SMOOTHING
        )

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, smooth_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} | Train Loss: {losses.avg}")
    return losses.avg


def evaluate(model, loader, device, use_tta=False):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The validation data loader.
        device (str): Device to run on.
        use_tta (bool): Whether to use Test Time Augmentation (Cite solution_lesson_node_00025).

    Returns:
        tuple: (average_loss, log_loss_metric, all_logits, all_targets)
               all_logits and all_targets are numpy arrays used for calibration.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).float().unsqueeze(1)

            if use_tta:
                # TTA: Original + HFlip + VFlip (Cite solution_lesson_node_00031)
                logits_sum = model(images, angles)
                logits_sum += model(torch.flip(images, [3]), angles)
                logits_sum += model(torch.flip(images, [2]), angles)
                logits = logits_sum / 3.0
            else:
                logits = model(images, angles)

            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Store logits and targets for metric calculation and calibration
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    if len(all_logits) > 0:
        all_logits = np.concatenate(all_logits).flatten()
        all_targets = np.concatenate(all_targets).flatten()
    else:
        all_logits = np.array([])
        all_targets = np.array([])

    # Calculate Competition Metric (Log Loss)
    # Convert logits to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    metric = calculate_log_loss(all_targets, probs)

    print(f"Validation Loss (BCE): {losses.avg}")
    print(f"Validation Log Loss: {metric}")

    return losses.avg, metric, all_logits, all_targets


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The test data loader.
        device (str): Device to run on.

    Returns:
        tuple: (ids, probabilities)
               ids is a numpy array of strings.
               probabilities is a numpy array of floats (0-1).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
    else:
        all_probs = np.array([])

    return np.array(all_ids), all_probs


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """

    def __init__(
        self, patience=Config.EARLY_STOPPING_PATIENCE, min_delta=0, mode="min"
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_worse(score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def _is_worse(self, score):
        if self.mode == "min":
            return score > self.best_score - self.min_delta
        else:
            return score < self.best_score + self.min_delta
