import torch
import numpy as np
from library.utils import calculate_map_score


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    """
    Performs one epoch of training with Automatic Mixed Precision (AMP) and Deep Supervision.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        scaler: The GradScaler for AMP.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)

            # Handle Deep Supervision (List of outputs)
            if isinstance(outputs, list):
                loss = 0.0
                # Apply Equal Weights (1.0) to all auxiliary heads
                for o in outputs:
                    loss += criterion(o, masks)
            else:
                loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs one epoch of validation with Test-Time Augmentation (TTA) and Cropping.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average Loss, Average mAP)
    """
    model.eval()
    running_loss = 0.0
    val_scores = []

    # Calculate cropping indices to revert 128x128 padding to 101x101
    # The dataset uses reflection padding which is typically centered or symmetric.
    target_size = 128
    orig_size = 101
    start_idx = (target_size - orig_size) // 2
    end_idx = start_idx + orig_size

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # 1. Forward Pass for Loss (Standard pass without TTA for consistent loss tracking)
            # In eval mode, the model returns a single tensor (the final head)
            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            # 2. Test-Time Augmentation (TTA) for Metrics
            # Original Input
            probs_normal = torch.sigmoid(logits)

            # Horizontal Flip Input
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # Revert Flip on Output
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average Predictions
            probs_avg = (probs_normal + probs_flip) / 2.0

            # 3. Crop to Original Size (101x101)
            # We must evaluate on the original signal region, removing the padded border.
            preds_cropped = probs_avg[:, :, start_idx:end_idx, start_idx:end_idx]
            masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

            # 4. Calculate mAP
            # calculate_map_score handles binarization at threshold 0.5 internally if not specified,
            # but we pass it explicitly for clarity. It computes mean AP over the batch.
            batch_score = calculate_map_score(
                preds_cropped, masks_cropped, decision_threshold=0.5
            )
            val_scores.append(batch_score)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_map = np.mean(val_scores)

    return epoch_loss, epoch_map
