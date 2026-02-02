import torch
import numpy as np
from library.config import CFG
from library.utils import calculate_roc_auc


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run computations on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # CrossEntropyLoss expects class indices (LongTensor), but dataset returns one-hot floats.
        # We convert one-hot to indices here.
        targets = torch.argmax(labels, dim=1)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step the scheduler after the epoch
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    print(f"Train Loss: {epoch_loss}")

    return epoch_loss


def valid_one_epoch(model, dataloader, criterion, device, use_tta=False):
    """
    Performs one epoch of validation.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run computations on.
        use_tta (bool): Whether to use Test-Time Augmentation.

    Returns:
        tuple: (average_loss, predictions, true_labels)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            targets = torch.argmax(labels, dim=1)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            # Cite Lesson 33: Validate exactly as you predict (TTA)
            if use_tta:
                # Horizontal Flip (dim 3 is width)
                images_hf = torch.flip(images, [3])
                outputs_hf = model(images_hf)
                probs_hf = torch.softmax(outputs_hf, dim=1)

                # Vertical Flip (dim 2 is height)
                images_vf = torch.flip(images, [2])
                outputs_vf = model(images_vf)
                probs_vf = torch.softmax(outputs_vf, dim=1)

                # Average predictions
                probs = (probs + probs_hf + probs_vf) / 3.0

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metric
    epoch_auc = calculate_roc_auc(all_labels, all_preds)

    print(f"Valid Loss: {epoch_loss}")
    print(f"Valid AUC: {epoch_auc}")

    return epoch_loss, all_preds, all_labels
