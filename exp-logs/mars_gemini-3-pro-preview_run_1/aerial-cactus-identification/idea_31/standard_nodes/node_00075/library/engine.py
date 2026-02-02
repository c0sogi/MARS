import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, update_bn
from library.config import DEVICE, MIXUP_ALPHA, AUX_LOSS_WEIGHT
from library.utils import calculate_roc_auc, sigmoid
from library.dataset import mixup_data


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup and Deep Supervision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        # Models return (main_out, aux_out) in training mode
        main_out, aux_out = model(images)

        # Reshape targets to match output (B, 1)
        targets_a = targets_a.view(-1, 1)
        targets_b = targets_b.view(-1, 1)

        # Calculate Main Loss
        loss_main = lam * criterion(main_out, targets_a) + (1 - lam) * criterion(
            main_out, targets_b
        )

        # Calculate Aux Loss (Deep Supervision)
        loss_aux = lam * criterion(aux_out, targets_a) + (1 - lam) * criterion(
            aux_out, targets_b
        )

        # Total Loss
        loss = loss_main + AUX_LOSS_WEIGHT * loss_aux

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            # Models return only main_out in eval mode
            outputs = model(images)

            # Calculate Loss
            loss = criterion(outputs, labels.view(-1, 1))
            running_loss += loss.item() * images.size(0)

            # Store predictions and targets for AUC
            preds = sigmoid(outputs.cpu().numpy())
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Metric
    auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, auc


def predict_tta(model, loader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Also handles RepVGG structural re-parameterization if applicable.
    """
    model.eval()

    # Handle RepVGG Re-parameterization (Fusion)
    # If model is wrapped in AveragedModel (SWA), access the underlying module
    inner_model = model.module if hasattr(model, "module") else model

    # Check if the model has the switch_to_deploy method (RepVGG variants)
    if hasattr(inner_model, "switch_to_deploy") and not inner_model.deploy:
        inner_model.switch_to_deploy()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # TTA Strategy: 4 Views
            # 1. Original
            out1 = model(images)

            # 2. Horizontal Flip
            out2 = model(torch.flip(images, [3]))

            # 3. Vertical Flip
            out3 = model(torch.flip(images, [2]))

            # 4. Rotate 180 (Horizontal + Vertical Flip)
            out4 = model(torch.flip(images, [2, 3]))

            # Average logits or probabilities?
            # Averaging probabilities is generally more stable for ensembles/TTA
            prob1 = sigmoid(out1.cpu().numpy())
            prob2 = sigmoid(out2.cpu().numpy())
            prob3 = sigmoid(out3.cpu().numpy())
            prob4 = sigmoid(out4.cpu().numpy())

            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            all_preds.append(avg_prob)
            all_ids.extend(ids)

    return np.concatenate(all_preds).flatten(), all_ids


def update_swa_bn(swa_model, loader, device):
    """
    Updates the Batch Normalization statistics for the SWA model.
    """
    # update_bn expects the loader to return just images or (images, targets)
    # Our loader returns (images, targets), which update_bn handles correctly.
    update_bn(loader, swa_model, device=device)
