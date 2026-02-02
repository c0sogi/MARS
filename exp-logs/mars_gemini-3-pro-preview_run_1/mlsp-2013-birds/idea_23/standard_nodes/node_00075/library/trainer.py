import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, calculate_roc_auc
from library.data import mixup_data

# Initialize logger
logger = get_logger("trainer")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Executes one epoch of training using SAM optimizer and Mixup.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (library.sam.SAM): The SAM optimizer instance.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        device (str): Device to run training on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # Iterate over dataloader
    # Using tqdm for progress tracking is optional but helpful,
    # prompt asked not to print progress bars, so we iterate directly.
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        # Note: mixup_data returns mixed_x, y_a, y_b, lam
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, Config.MIXUP_ALPHA, device
        )

        # ==========================
        # SAM Step 1: Gradient Ascent
        # ==========================
        # Forward pass
        outputs = model(inputs)

        # Compute loss with Mixup
        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (
            1 - lam
        )

        # Backward pass to compute gradients
        loss.backward()

        # Perturb weights (ascent) and clear gradients
        optimizer.first_step(zero_grad=True)

        # ==========================
        # SAM Step 2: Gradient Descent
        # ==========================
        # Forward pass at perturbed state
        outputs_adv = model(inputs)

        # Compute loss again at perturbed state
        loss_adv = criterion(outputs_adv, targets_a) * lam + criterion(
            outputs_adv, targets_b
        ) * (1 - lam)

        # Backward pass at perturbed state
        loss_adv.backward()

        # Restore weights and update using perturbed gradients
        optimizer.second_step(zero_grad=True)

        # Track metrics
        # We track the first loss (original state) for reporting
        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            num_batches += 1

            # Store predictions and targets for AUC
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)

        # Calculate ROC AUC
        auc_score = calculate_roc_auc(all_targets, all_probs)
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def update_swa_model(swa_model, model):
    """
    Updates the SWA (Stochastic Weight Averaging) model with parameters from the current model.

    Args:
        swa_model (torch.optim.swa_utils.AveragedModel): The SWA model wrapper.
        model (torch.nn.Module): The current training model.
    """
    swa_model.update_parameters(model)


def finalize_swa(swa_model, train_loader, device):
    """
    Updates the BatchNorm statistics of the SWA model using the training data.
    This is necessary because SWA averages weights but not BN statistics.

    Args:
        swa_model (torch.optim.swa_utils.AveragedModel): The SWA model.
        train_loader (DataLoader): The training data loader.
        device (str): Device to run on.
    """
    logger.info("Updating SWA BatchNorm statistics...")
    # update_bn expects the model, loader, and device
    update_bn(train_loader, swa_model, device=device)
