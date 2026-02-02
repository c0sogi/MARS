import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calc_log_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The training dataloader.
        device (str): The device to run training on.
        epoch (int): The current epoch number.
        scheduler (object, optional): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    total_loss = 0.0
    num_samples = 0

    # BCEWithLogitsLoss combines Sigmoid layer and the BCELoss in one single class.
    # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Outputs shape is (Batch_Size, 1), Targets shape is (Batch_Size,)
        # Squeeze outputs to match targets
        loss = criterion(outputs.squeeze(1), targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss (multiply by batch size to handle potential last incomplete batch correctly)
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

        # Note: Scheduler step is typically handled after the epoch for CosineAnnealingLR
        # or similar epoch-based schedulers.

    avg_loss = total_loss / num_samples
    print(f"Epoch {epoch} Training Loss: {avg_loss}")

    return avg_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The validation dataloader.
        device (str): The device to run evaluation on.

    Returns:
        tuple: (average_loss, log_loss_score)
    """
    model.eval()

    total_loss = 0.0
    num_samples = 0

    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            logits = outputs.squeeze(1)

            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            # Convert logits to probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = total_loss / num_samples

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate Log Loss using the utility function
    metric_score = calc_log_loss(all_targets, all_preds)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Log Loss: {metric_score}")

    return avg_loss, metric_score


def inference_fn(model, data_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA involves averaging predictions from the original image and a horizontally flipped version.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): The test dataloader.
        device (str): The device to run inference on.

    Returns:
        tuple: (ids, predictions) where ids is a list of image IDs and predictions is a list of probabilities.
    """
    model.eval()

    ids = []
    preds = []

    with torch.no_grad():
        for images, img_ids in data_loader:
            images = images.to(device)

            # 1. Forward pass with original images
            output_orig = model(images)
            probs_orig = torch.sigmoid(output_orig)

            # 2. Forward pass with horizontally flipped images (TTA)
            # Image tensor format is (B, C, H, W). Flip on dimension 3 (W).
            images_flip = torch.flip(images, dims=[3])
            output_flip = model(images_flip)
            probs_flip = torch.sigmoid(output_flip)

            # 3. Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            # Store results
            ids.extend(img_ids.numpy())
            # Flatten to ensure we have a 1D array of probabilities
            preds.extend(avg_probs.cpu().numpy().flatten())

    return ids, preds
