import numpy as np
import torch
from library.utils import get_score
from library.losses import DistillationLoss


def train_one_epoch(model, loader, optimizer, device, epoch, loss_fn, mixup_alpha=0.4):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (str): 'cuda' or 'cpu'.
        epoch (int): Current epoch number.
        loss_fn (nn.Module): Loss function (WeightedBCELoss or DistillationLoss).
        mixup_alpha (float): Alpha parameter for Mixup. Set to 0 to disable.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets, soft_targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        soft_targets = soft_targets.to(device)

        batch_size = images.size(0)

        # Apply Mixup if enabled
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]
            mixed_soft_targets = lam * soft_targets + (1 - lam) * soft_targets[index]

            outputs = model(mixed_images)

            # Calculate loss based on loss function type
            if isinstance(loss_fn, DistillationLoss):
                # Distillation requires student logits, teacher logits (soft targets), and hard targets
                loss = loss_fn(outputs, mixed_soft_targets, mixed_targets)
            else:
                # Standard training uses only hard targets
                loss = loss_fn(outputs, mixed_targets)

        else:
            outputs = model(images)

            if isinstance(loss_fn, DistillationLoss):
                loss = loss_fn(outputs, soft_targets, targets)
            else:
                loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f}")

    return epoch_loss


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        device (str): 'cuda' or 'cpu'.
        loss_fn (nn.Module): Loss function.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, soft_targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            soft_targets = soft_targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)

            if isinstance(loss_fn, DistillationLoss):
                loss = loss_fn(outputs, soft_targets, targets)
            else:
                loss = loss_fn(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for scoring
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = get_score(all_targets, all_preds)

    print(f"Validation | Loss: {epoch_loss:.6f} | AUC: {score:.6f}")

    return epoch_loss, score


def inference(model, loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).

    TTA Strategy: Average predictions of Original, Roll-25%, Roll-50%, and Roll-75%.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (str): 'cuda' or 'cpu'.

    Returns:
        np.ndarray: Predicted probabilities of shape (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            # Shape: (Batch, Channels, Height, Width)
            batch_size, _, _, w = images.shape

            # Define shifts for TTA: 0%, 25%, 50%, 75% of width
            shifts = [0, w // 4, w // 2, 3 * w // 4]
            batch_probs_list = []

            for s in shifts:
                if s == 0:
                    inputs = images
                else:
                    # Roll along width dimension (dim 3)
                    inputs = torch.roll(images, shifts=s, dims=3)

                logits = model(inputs)
                probs = torch.sigmoid(logits)
                batch_probs_list.append(probs)

            # Average probabilities across the 4 TTA variants
            avg_probs = torch.stack(batch_probs_list).mean(dim=0)
            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
