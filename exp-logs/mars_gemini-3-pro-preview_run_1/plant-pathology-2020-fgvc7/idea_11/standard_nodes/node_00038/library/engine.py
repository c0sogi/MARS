import torch
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

        # Note: For CosineAnnealingWarmRestarts with T_0 in epochs,
        # we typically step after the epoch, not per batch.
        # However, if a batch-level scheduler were passed, it would be stepped here.
        # We leave the scheduler step to the main loop or end of epoch based on Config.

    epoch_loss = running_loss / dataset_size

    # Step scheduler at the end of the epoch if provided and it's not a batch-level scheduler
    # (Assuming standard usage for the Config provided)
    if scheduler is not None:
        scheduler.step()

    return epoch_loss


def valid_one_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use for evaluation.

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply softmax to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def inference_fn(model, dataloader, device, use_tta=False):
    """
    Generates predictions for the test set, optionally using Test-Time Augmentation (TTA).

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The test dataloader.
        device (torch.device): The device to use for inference.
        use_tta (bool): Whether to apply TTA (Horizontal and Vertical Flips).

    Returns:
        np.ndarray: Predicted probabilities of shape (N_samples, N_classes).
    """
    model.eval()
    final_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Original Prediction
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # 2. Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                outputs_h = model(images_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # 3. Vertical Flip
                images_v = torch.flip(images, dims=[2])
                outputs_v = model(images_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average predictions
                probs = (probs + probs_h + probs_v) / 3.0

            final_preds.append(probs.cpu().numpy())

    return np.concatenate(final_preds)
