import torch
import numpy as np
from library import config, utils


def train_one_epoch(model, loader, optimizer, device, loss_fn, scheduler=None):
    """
    Trains the model for one epoch.
    Handles Mixup augmentation and Distillation loss.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        loss_fn: Instance of DistillationLoss.
        scheduler: Optional learning rate scheduler (stepped per batch).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = utils.AverageMeter()

    # Enable/Disable Mixup based on config
    use_mixup = config.MIXUP_ALPHA > 0

    for batch_idx, data in enumerate(loader):
        images = data["image"].to(device)
        targets = data["target"].to(device)
        soft_targets = data["soft_target"].to(device)

        batch_size = images.size(0)

        if use_mixup:
            # Generate Mixup lambda
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            # Mix inputs
            mixed_images = lam * images + (1 - lam) * images[index]

            # Mix hard targets
            mixed_targets = lam * targets + (1 - lam) * targets[index]

            # Mix soft targets (essential for Stage 2 Distillation)
            # If in Stage 1, soft_targets are zeros, so this remains zero.
            mixed_soft_targets = lam * soft_targets + (1 - lam) * soft_targets[index]

            # Forward pass
            logits = model(mixed_images)

            # Compute Loss
            loss = loss_fn(logits, mixed_targets, teacher_probs=mixed_soft_targets)
        else:
            # Standard forward pass
            logits = model(images)
            loss = loss_fn(logits, targets, teacher_probs=soft_targets)

        losses.update(loss.item(), batch_size)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return losses.avg


def valid_one_epoch(model, loader, device, loss_fn):
    """
    Validates the model on the validation set.
    Computes Loss (Hard only) and AUC.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.
        loss_fn: Instance of DistillationLoss.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    losses = utils.AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            targets = data["target"].to(device)

            batch_size = images.size(0)

            logits = model(images)

            # For validation, we measure performance against Ground Truth (Hard Labels).
            # Passing teacher_probs=None forces the DistillationLoss to compute only the hard loss component.
            loss = loss_fn(logits, targets, teacher_probs=None)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Compute AUC
    auc = utils.get_score(all_targets, all_preds)

    return losses.avg, auc


def predict(model, loader, device):
    """
    Performs inference on the test set with Test-Time Augmentation (TTA).
    Strategy: Original + 3 Time-Roll shifts (0%, 25%, 50%, 75% of width).

    Args:
        model: The PyTorch model.
        loader: DataLoader for test data.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary mapping rec_id (int) to predicted probabilities (np.array).
    """
    model.eval()

    predictions = {}

    # Define TTA shifts as fractions of the image width (Time axis)
    shifts_ratios = [0.0, 0.25, 0.50, 0.75]

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            # rec_id comes as a tensor from the loader
            rec_ids = data["rec_id"].numpy()

            # images shape: [Batch, Channels, Height, Width]
            batch_size, _, _, width = images.shape

            batch_probs_variants = []

            # TTA Loop: Predict on original and shifted versions
            for ratio in shifts_ratios:
                shift_pixels = int(width * ratio)

                if shift_pixels == 0:
                    input_images = images
                else:
                    # Roll along the width dimension (dim 3)
                    input_images = torch.roll(images, shifts=shift_pixels, dims=3)

                logits = model(input_images)
                probs = torch.sigmoid(logits)
                batch_probs_variants.append(probs.cpu().numpy())

            # Average predictions across the 4 TTA variants
            # Stack shape: [4, Batch, Num_Classes] -> Mean -> [Batch, Num_Classes]
            avg_probs = np.mean(np.stack(batch_probs_variants), axis=0)

            # Store results keyed by rec_id
            for i, rec_id in enumerate(rec_ids):
                predictions[rec_id] = avg_probs[i]

    return predictions
