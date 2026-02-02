import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_roc_auc


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to run training on ('cpu' or 'cuda').
        epoch (int): Current epoch number.

    Returns:
        dict: Dictionary containing average losses for the epoch.
    """
    model.train()

    # Meters to track losses
    losses_total = AverageMeter()
    losses_primary = AverageMeter()
    losses_aux = AverageMeter()

    # Define Loss Functions
    # Primary: Binary Cross Entropy with Logits (Weighted for class imbalance)
    # We wrap POS_WEIGHT in a tensor and move to device
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion_primary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Auxiliary: Cross Entropy Loss for multi-class diagnosis
    criterion_aux = nn.CrossEntropyLoss()

    for step, batch in enumerate(dataloader):
        # Move data to device
        images = batch["image"].to(device)
        meta = batch["meta"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Shape (B, 1)
        aux_targets = batch["aux_target"].to(device)  # Shape (B,)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits_malignancy, logits_diagnosis = model(images, meta)

        # Calculate Losses
        loss_primary = criterion_primary(logits_malignancy, targets)
        loss_aux = criterion_aux(logits_diagnosis, aux_targets)

        # Weighted Sum
        loss = loss_primary + (Config.AUX_LOSS_WEIGHT * loss_aux)

        # Backward pass
        loss.backward()

        # Gradient Clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Optimizer Step
        optimizer.step()

        # Update Meters
        batch_size = images.size(0)
        losses_total.update(loss.item(), batch_size)
        losses_primary.update(loss_primary.item(), batch_size)
        losses_aux.update(loss_aux.item(), batch_size)

    # Print metrics for the epoch
    print(f"Epoch {epoch} Train Summary:")
    print(f"  Total Loss: {losses_total.avg}")
    print(f"  Primary Loss: {losses_primary.avg}")
    print(f"  Aux Loss: {losses_aux.avg}")

    return {
        "loss_total": losses_total.avg,
        "loss_primary": losses_primary.avg,
        "loss_aux": losses_aux.avg,
    }


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (avg_loss, auc_score)
    """
    model.eval()

    losses_total = AverageMeter()

    # Define Loss Functions (same as training to track validation loss)
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion_primary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            aux_targets = batch["aux_target"].to(device)

            # Forward pass (Original)
            logits_mal_1, logits_diag_1 = model(images, meta)

            # Calculate Loss (on original images only)
            loss_primary = criterion_primary(logits_mal_1, targets)
            loss_aux = criterion_aux(logits_diag_1, aux_targets)
            loss = loss_primary + (Config.AUX_LOSS_WEIGHT * loss_aux)
            losses_total.update(loss.item(), images.size(0))

            # TTA: 4-View (Original, H-Flip, V-Flip, HV-Flip)
            # 1. Original
            preds_1 = torch.sigmoid(logits_mal_1)

            # 2. Horizontal Flip
            logits_mal_2, _ = model(torch.flip(images, [3]), meta)
            preds_2 = torch.sigmoid(logits_mal_2)

            # 3. Vertical Flip
            logits_mal_3, _ = model(torch.flip(images, [2]), meta)
            preds_3 = torch.sigmoid(logits_mal_3)

            # 4. H+V Flip
            logits_mal_4, _ = model(torch.flip(images, [2, 3]), meta)
            preds_4 = torch.sigmoid(logits_mal_4)

            # Average Predictions
            preds_avg = (preds_1 + preds_2 + preds_3 + preds_4) / 4.0

            # Store predictions and targets for AUC calculation
            all_preds.append(preds_avg.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate ROC AUC
    auc_score = get_roc_auc(all_targets, all_preds)

    print(f"Validation Summary:")
    print(f"  Loss: {losses_total.avg}")
    print(f"  AUC: {auc_score}")

    return losses_total.avg, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for test data.
        device (str): Device to run prediction on.

    Returns:
        np.array: Array of predicted probabilities for the malignant class.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            # TTA: 4-View (Original, H-Flip, V-Flip, HV-Flip)
            # 1. Original
            logits_1, _ = model(images, meta)
            preds_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip
            logits_2, _ = model(torch.flip(images, [3]), meta)
            preds_2 = torch.sigmoid(logits_2)

            # 3. Vertical Flip
            logits_3, _ = model(torch.flip(images, [2]), meta)
            preds_3 = torch.sigmoid(logits_3)

            # 4. H+V Flip
            logits_4, _ = model(torch.flip(images, [2, 3]), meta)
            preds_4 = torch.sigmoid(logits_4)

            # Average
            preds_avg = (preds_1 + preds_2 + preds_3 + preds_4) / 4.0
            all_preds.append(preds_avg.cpu().numpy())

    return np.concatenate(all_preds).ravel()
