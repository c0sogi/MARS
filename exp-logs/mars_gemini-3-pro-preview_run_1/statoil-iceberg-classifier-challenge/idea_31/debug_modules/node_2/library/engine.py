import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Performs one epoch of standard training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch
        if len(batch) == 3:
            images, angles, targets = batch
        else:
            raise ValueError("Training loader must provide labels.")

        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    # Step scheduler if provided and not dependent on metrics (e.g., ReduceLROnPlateau)
    if scheduler is not None:
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def train_swa_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training during the SWA phase.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        if len(batch) == 3:
            images, angles, targets = batch
        else:
            raise ValueError("Training loader must provide labels.")

        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate_tta(model, loader, criterion, device):
    """
    Evaluates the model using Exhaustive Closed-Group TTA (4 views).
    Returns Log Loss and Accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, angles, targets = batch
            else:
                raise ValueError("Validation loader must provide labels.")

            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            # TTA: Original, HFlip, VFlip, Rot180
            # 1. Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip (dim 3)
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip (dim 2)
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # 4. Rotate 180 (dim 2 and 3)
            images_r = torch.flip(images, [2, 3])
            out4 = model(images_r, angles)
            prob4 = torch.sigmoid(out4)

            # Average probabilities
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Metrics
    # Clip predictions to avoid log(0)
    preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
    score_log_loss = log_loss(all_targets, preds_clipped)

    preds_binary = (all_preds > 0.5).astype(int)
    score_acc = accuracy_score(all_targets, preds_binary)

    return score_log_loss, score_acc


def predict_tta(model, loader, device):
    """
    Generates predictions for the test set using TTA.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Test loader typically returns (images, angles)
            if len(batch) == 2:
                images, angles = batch
            elif len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch[0], batch[1]

            images = images.to(device)
            angles = angles.to(device)

            # TTA Steps
            out1 = torch.sigmoid(model(images, angles))
            out2 = torch.sigmoid(model(torch.flip(images, [3]), angles))
            out3 = torch.sigmoid(model(torch.flip(images, [2]), angles))
            out4 = torch.sigmoid(model(torch.flip(images, [2, 3]), angles))

            avg_prob = (out1 + out2 + out3 + out4) / 4.0
            all_preds.append(avg_prob.cpu().numpy())

    return np.concatenate(all_preds)
