import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, data_loader, optimizer, device, criterion, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run on.
        criterion (torch.nn.Module): The loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Targets from dataset are (Batch, 4) float tensors (effectively one-hot/soft).
        # CrossEntropyLoss expects class indices (Batch,).
        # We convert using argmax, assuming single-label ground truth.
        loss = criterion(outputs, torch.argmax(targets, dim=1))

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    # Step the scheduler at the end of the epoch (strictly synchronized with epochs)
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / total_samples
    print(f"Train Loss: {epoch_loss}")

    return epoch_loss


def validate_one_epoch(model, data_loader, device, criterion, scheduler=None):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run on.
        criterion (torch.nn.Module): The loss function.
        scheduler (optional): Accepted for API consistency but not used in validation.

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, torch.argmax(targets, dim=1))

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Apply softmax to get probabilities for ROC AUC
            probs = torch.softmax(outputs, dim=1)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    epoch_loss = running_loss / total_samples

    # Concatenate all batches
    preds_array = np.vstack(preds_list)
    targets_array = np.vstack(targets_list)

    # Calculate Mean Column-wise ROC AUC
    # 'macro' average computes AUC for each class and takes the unweighted mean
    # 'ovr' (One-vs-Rest) is appropriate for multi-class/multi-label AUC
    try:
        val_auc = roc_auc_score(
            targets_array, preds_array, average="macro", multi_class="ovr"
        )
    except Exception as e:
        # Fallback for edge cases (e.g., only one class present in small validation set)
        print(f"Warning: ROC AUC calculation failed: {e}")
        val_auc = 0.0

    print(f"Val Loss: {epoch_loss}")
    print(f"Val AUC: {val_auc}")

    return epoch_loss, val_auc
