import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.data import mixup_data
from library.utils import compute_robust_auc


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    """
    Trains the model for one epoch using Mixup and Lookahead optimizer.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup if enabled
        if Config.ENABLE_MIXUP:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, alpha=Config.MIXUP_ALPHA, use_cuda=(device != "cpu")
            )
            images, targets_a, targets_b = map(
                torch.autograd.Variable, (images, targets_a, targets_b)
            )

            outputs = model(images)
            loss = criterion(outputs, targets_a) * lam + criterion(
                outputs, targets_b
            ) * (1 - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the Robust AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Robust AUC
    auc_score = compute_robust_auc(all_targets, all_preds)

    return avg_loss, auc_score


def inference_with_tta(model, dataloader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Strategy: Average predictions of [Original, Left Shift, Right Shift].
    Shift magnitude is restricted to 10% of width with zero-padding.
    """
    model.eval()
    all_preds = []

    # Calculate shift amount (10% of width)
    # Assuming images are (B, C, H, W)
    # Config.IMG_SIZE is (H, W) -> (224, 224)
    width = Config.IMG_SIZE[1]
    shift_pixels = int(width * Config.SHIFT_LIMIT)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Right Shift (Time Delay)
            # Move pixels to right, pad left with zeros
            img_right = torch.zeros_like(images)
            img_right[:, :, :, shift_pixels:] = images[:, :, :, :-shift_pixels]
            out_right = model(img_right)
            prob_right = torch.sigmoid(out_right)

            # 3. Left Shift (Time Advance)
            # Move pixels to left, pad right with zeros
            img_left = torch.zeros_like(images)
            img_left[:, :, :, :-shift_pixels] = images[:, :, :, shift_pixels:]
            out_left = model(img_left)
            prob_left = torch.sigmoid(out_left)

            # Average Predictions
            avg_prob = (prob_orig + prob_right + prob_left) / 3.0

            all_preds.append(avg_prob.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
