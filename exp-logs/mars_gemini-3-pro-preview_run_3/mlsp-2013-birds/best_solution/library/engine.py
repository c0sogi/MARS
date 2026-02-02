import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from library.utils import RobustMetric
from library.config import Config


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Calibrated Mixup: mixes x and y with a lambda sampled from Beta(alpha, alpha).
    Enforces lambda >= 0.5 to bias towards the original sample (ground truth).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Calibrated Mixup: bias towards the primary sample
    lam = max(lam, 1 - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    input_key="image",
    mixup_alpha=Config.MIXUP_ALPHA,
):
    """
    Training loop for one epoch.
    Supports both Image (CNN) and Feature (MLP) training via input_key.
    Applies Calibrated Mixup to inputs.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Iterate over dataloader
    # Note: tqdm is not used to keep output clean as per requirements,
    # but the loop structure is standard.
    for batch_idx, batch in enumerate(dataloader):
        # Extract data based on model type (image vs features)
        inputs = batch[input_key].to(device)
        targets = batch["target"].to(device)

        batch_size = inputs.size(0)

        # Apply Mixup
        if mixup_alpha > 0:
            inputs, targets, _ = mixup_data(
                inputs, targets, alpha=mixup_alpha, device=device
            )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device, input_key="image"):
    """
    Validation loop for one epoch.
    Computes Loss and AUROC using RobustMetric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    metric = RobustMetric()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[input_key].to(device)
            targets = batch["target"].to(device)

            batch_size = inputs.size(0)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update metrics
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Update RobustMetric (expects sigmoid probabilities or logits?
            # RobustMetric uses roc_auc_score which handles logits fine for ranking,
            # but usually probabilities are safer. However, utils.py says "logits are acceptable".
            # We pass logits directly.)
            metric.update(outputs, targets)

    epoch_loss = running_loss / dataset_size
    epoch_auc = metric.compute()

    return epoch_loss, epoch_auc
