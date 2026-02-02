import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter


def train_one_epoch(epoch, model, loader, optimizer, criterion, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to train on (cuda/cpu).
        scheduler (LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    # Step the scheduler if provided (typically for Cosine Annealing per epoch)
    if scheduler is not None:
        scheduler.step()

    # Print metrics with full precision as requested
    print(f"Epoch {epoch} Training Loss: {losses.avg}")

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, accuracy, all_logits, all_labels)
               all_logits and all_labels are returned for Post-Hoc Calibration.
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Store logits and labels for calibration
            logits_list.append(outputs.cpu())
            labels_list.append(targets.cpu())

    accuracy = 100.0 * correct / total

    # Print metrics with full precision
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Accuracy: {accuracy}")

    # Concatenate all logits and labels
    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return losses.avg, accuracy, all_logits, all_labels


def update_swa(swa_model, model, swa_scheduler, epoch):
    """
    Updates the SWA model parameters and scheduler if the current epoch
    is within the SWA phase.

    Args:
        swa_model (AveragedModel): The SWA model wrapper.
        model (nn.Module): The current training model.
        swa_scheduler (LRScheduler): The SWA specific scheduler.
        epoch (int): Current epoch number.
    """
    if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
        swa_model.update_parameters(model)
        if swa_scheduler is not None:
            swa_scheduler.step()


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    """

    def __init__(self, patience=5, min_delta=0.0, mode="min"):
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
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "min":
                improvement = self.best_score - score
            else:
                improvement = score - self.best_score

            if improvement > self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
