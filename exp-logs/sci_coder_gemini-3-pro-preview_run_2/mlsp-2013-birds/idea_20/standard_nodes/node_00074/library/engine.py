import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_roc_auc
from library.losses import SiameseConsistencyLoss


def train_one_epoch(model, loader, optimizer, device, pos_weights):
    """
    Executes one training epoch with Siamese Temporal Consistency Regularization.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to run training on ('cuda' or 'cpu').
        pos_weights (torch.Tensor): Class weights for imbalance handling.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Initialize the custom Siamese loss
    criterion = SiameseConsistencyLoss(
        pos_weights=pos_weights, consistency_lambda=Config.CONSISTENCY_LAMBDA
    )

    for images, labels in loader:
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # --- Mixup Augmentation ---
        # Sample lambda from Beta distribution
        if Config.MIXUP_ALPHA > 0:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
        else:
            lam = 1.0

        # Shuffle indices for mixing
        index = torch.randperm(batch_size).to(device)

        # Create mixed inputs and targets
        # Note: images already have SpecAugment applied via Dataset transforms
        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]

        # --- Siamese View Generation ---
        # Generate x_roll by applying a random cyclic time-shift
        # Dimension 3 is Width (Time axis)
        width = mixed_images.shape[3]
        shift = np.random.randint(0, width)
        images_roll = torch.roll(mixed_images, shifts=shift, dims=3)

        # --- Forward Pass ---
        optimizer.zero_grad()

        # Compute logits for both original (mixed) and rolled (mixed) views
        logits = model(mixed_images)
        logits_roll = model(images_roll)

        # --- Loss Calculation ---
        # Computes BCE for both views + MSE consistency term
        loss = criterion(logits, logits_roll, mixed_labels)

        # --- Backward Pass ---
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device, pos_weights):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Validation data loader.
        device (str): Device to run evaluation on.
        pos_weights (torch.Tensor): Class weights for imbalance handling.

    Returns:
        tuple: (average_loss, macro_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Use standard BCEWithLogitsLoss for validation metrics
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    with torch.no_grad():
        for images, labels in loader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Compute ROC AUC
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_auc = compute_roc_auc(all_targets, all_preds)
    else:
        val_auc = 0.0

    return val_loss, val_auc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Cyclic Rolling).
    Variants: Original, Roll 25%, Roll 50%, Roll 75%.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Test data loader.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Averaged predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # Shape: (Batch, Channels, Height, Width)
            width = images.shape[3]

            # Define TTA shifts: 0%, 25%, 50%, 75%
            shifts = [0, width // 4, width // 2, (3 * width) // 4]

            batch_probs_list = []

            for shift in shifts:
                if shift == 0:
                    img_variant = images
                else:
                    img_variant = torch.roll(images, shifts=shift, dims=3)

                logits = model(img_variant)
                probs = torch.sigmoid(logits)
                batch_probs_list.append(probs)

            # Stack along a new dimension (4, Batch, Num_Classes)
            stacked_probs = torch.stack(batch_probs_list, dim=0)

            # Average predictions across TTA variants
            avg_probs = torch.mean(stacked_probs, dim=0)

            all_preds.append(avg_probs.cpu().numpy())

    if len(all_preds) > 0:
        return np.concatenate(all_preds, axis=0)
    else:
        return np.array([])
