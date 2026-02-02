import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score
from torch.optim.swa_utils import AveragedModel

from library.config import Config


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function (e.g., BCEWithLogitsLoss).
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number (for logging).

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds

    print(f"Epoch {epoch} Train - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.6f}")

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, tta=True):
    """
    Evaluates the model, optionally using TTA.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        device (torch.device): Device to run evaluation on.
        tta (bool): Whether to use Test-Time Augmentation.

    Returns:
        tuple: (log_loss_score, accuracy_score)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)

            # Store labels
            all_labels.extend(labels.numpy())

            if tta:
                # 1. Original
                logits_orig = model(images, angles)
                probs_orig = torch.sigmoid(logits_orig)

                # 2. Horizontal Flip (dim 3 is width)
                images_h = torch.flip(images, [3])
                logits_h = model(images_h, angles)
                probs_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip (dim 2 is height)
                images_v = torch.flip(images, [2])
                logits_v = model(images_v, angles)
                probs_v = torch.sigmoid(logits_v)

                # Average probabilities
                avg_probs = (probs_orig + probs_h + probs_v) / 3.0
                all_probs.extend(avg_probs.cpu().numpy().flatten())

            else:
                # Standard evaluation
                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())

    # Compute metrics over the entire dataset
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Clip probabilities to avoid log(0) errors in log_loss if not handled by library
    all_probs = np.clip(all_probs, 1e-15, 1 - 1e-15)

    val_loss = log_loss(all_labels, all_probs)
    val_acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))

    print(f"Val - LogLoss: {val_loss:.10f}, Acc: {val_acc:.10f}")

    return val_loss, val_acc


def update_swa(swa_model, model):
    """Updates the SWA model parameters."""
    swa_model.update_parameters(model)


def update_bn_custom(loader, model, device):
    """
    Custom implementation of update_bn to handle:
    1. Dual inputs (image, angle)
    2. Loader returning (image, angle, label) tuples
    """
    model.train()
    model.to(device)

    # Reset running stats
    # Note: We assume the model is a standard PyTorch model where we can traverse modules
    # or simply calling .train() and forward passes updates them if momentum is set.
    # However, standard update_bn resets momentum to None to calculate simple average.
    # We will mimic the behavior of torch.optim.swa_utils.update_bn manually.

    # 1. Reset stats
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use simple average

    # 2. Forward pass to accumulate stats
    with torch.no_grad():
        for batch in loader:
            # Handle variable unpacking based on phase (train/test)
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)

            model(images, angles)

    # 3. Restore momentum (optional, but good practice if model is used further)
    # Standard SWA doesn't strictly require restoring momentum if only used for inference.
