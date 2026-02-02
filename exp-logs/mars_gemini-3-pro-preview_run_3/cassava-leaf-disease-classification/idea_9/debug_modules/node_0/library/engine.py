import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from library.config import Config
from library.utils import AverageMeter
from library.data import mixup_cutmix_fn


def train_one_epoch(epoch, model, train_loader, optimizer, scaler, device):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        scaler (GradScaler): Gradient scaler for AMP.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Loss function with label smoothing
    # Note: For MixUp/CutMix, we manually blend the losses, but the base criterion
    # handles the label smoothing on the targets (indices).
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing).to(device)

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp or CutMix
        # Returns mixed images and both sets of labels with the mixing lambda
        mixed_images, target_a, target_b, lam = mixup_cutmix_fn(
            images, labels, alpha=1.0, prob=0.5
        )

        optimizer.zero_grad()

        # Automatic Mixed Precision Forward Pass
        with autocast():
            outputs = model(mixed_images)
            # Calculate loss as weighted sum of losses for both targets
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(
                outputs, target_b
            )

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Train Epoch: {epoch} | Loss: {loss_meter.avg}")
    return loss_meter.avg


def valid_one_epoch(epoch, model, val_loader, device):
    """
    Validates the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to validate on.

    Returns:
        tuple: (Average Loss, Average Accuracy)
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing).to(device)

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

    # Print full precision metrics
    print(f"Valid Epoch: {epoch} | Loss: {loss_meter.avg} | Accuracy: {acc_meter.avg}")
    return loss_meter.avg, acc_meter.avg


class EarlyStopping:
    """
    Early Stopping helper class to stop training when validation metric doesn't improve.
    """

    def __init__(self, patience=Config.patience, min_delta=0, mode="max"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation score improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "max":
                improve = score > (self.best_score + self.min_delta)
            else:
                improve = score < (self.best_score - self.min_delta)

            if improve:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
