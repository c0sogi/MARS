import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_log_loss


class DistillationLoss(nn.Module):
    """
    Custom Loss function for Knowledge Distillation.
    Combines BCE Loss on hard labels with BCE Loss on soft teacher targets.
    Formula: L = alpha * BCE(hard) + (1 - alpha) * BCE(soft)
    """

    def __init__(self, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, soft_targets=None):
        """
        Args:
            logits: Model output logits [B, 1]
            targets: Ground truth binary labels [B] or [B, 1]
            soft_targets: Teacher probability predictions [B] or [B, 1]
        """
        # Ensure targets match logits shape [B, 1]
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        loss_hard = self.bce(logits, targets)

        if soft_targets is not None:
            if soft_targets.dim() == 1:
                soft_targets = soft_targets.view(-1, 1)
            # BCEWithLogitsLoss works for soft targets (probabilities) as well
            # It computes -[y * log(sigma(x)) + (1-y) * log(1-sigma(x))]
            loss_soft = self.bce(logits, soft_targets)
            return self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft

        return loss_hard


def train_one_epoch(model, loader, optimizer, device, epoch, use_distillation=False):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
        epoch: Current epoch number (for printing)
        use_distillation: Boolean, whether to use soft targets

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter()
    criterion = DistillationLoss(alpha=Config.DISTILLATION_ALPHA)

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

        soft_targets = None
        if use_distillation and "soft_target" in batch:
            soft_targets = batch["soft_target"].to(device)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, labels, soft_targets)

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
