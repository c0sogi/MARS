import torch
import numpy as np
import sys
from library.utils import compute_metric
from library.dataset import mixup_data


def train_one_epoch(model, optimizer, scheduler, dataloader, device, criterion):
    """
    Trains the model for one epoch using Mixup and Asymmetric Loss.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler (optional).
        dataloader (torch.utils.data.DataLoader): Training dataloader.
        device (torch.device): Device to run training on.
        criterion (nn.Module): Loss function (AsymmetricLoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=0.4, device=device
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)

        # Compute loss (Mixup strategy: mix the losses)
        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (
            1 - lam
        )

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            # Step scheduler if it's batch-based, otherwise it might be stepped per epoch outside
            # Given "Constant Learning Rate", scheduler might be None or a dummy.
            # We assume standard PyTorch usage where batch-level schedulers (like OneCycle)
            # are stepped here, but if it's None, we skip.
            # Ideally, for constant LR, no scheduler is passed.
            pass

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metric
    auc_score = compute_metric(all_targets, all_preds)

    # Print full precision metric
    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation AUC: {auc_score}")

    return epoch_loss, auc_score


def inference_fn(model, dataloader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from: Original, Time-Roll(25%), Time-Roll(50%), Time-Roll(75%).

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Final averaged predictions (N_samples, N_classes).
    """
    model.eval()
    final_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Dimensions: (B, C, H, W)
            # W is the time axis (dim 3)
            width = images.shape[3]

            # TTA Shifts: 0, 25%, 50%, 75%
            shifts = [0, int(width * 0.25), int(width * 0.50), int(width * 0.75)]

            batch_tta_preds = []

            for shift in shifts:
                if shift == 0:
                    input_img = images
                else:
                    # Circular shift along the time axis
                    input_img = torch.roll(images, shifts=shift, dims=3)

                logits = model(input_img)
                probs = torch.sigmoid(logits)
                batch_tta_preds.append(probs)

            # Stack and average across the TTA dimension (dim 0 of the stack)
            # Shape of stack: (4, B, num_classes) -> Mean -> (B, num_classes)
            avg_batch_preds = torch.stack(batch_tta_preds).mean(dim=0)

            final_preds.append(avg_batch_preds.cpu().numpy())

    return np.concatenate(final_preds, axis=0)
