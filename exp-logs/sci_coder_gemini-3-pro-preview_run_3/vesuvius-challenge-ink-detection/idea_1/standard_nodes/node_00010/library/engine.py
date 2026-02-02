import os
import torch
import torch.nn as nn
from library.utils import fbeta_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for segmentation.

    The loss is calculated as: alpha * BCE + (1 - alpha) * DiceLoss
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight assigned to BCE loss. (1 - bce_weight) is assigned to Dice.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted probabilities (after Sigmoid), shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, 1, H, W).
        """
        # 1. BCE Loss
        bce = self.bce_loss(preds, targets)

        # 2. Dice Loss (Soft Dice for differentiability)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Combined Loss
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Torch optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for volumes, labels in dataloader:
        volumes = volumes.to(device)
        labels = labels.to(device)
        batch_size = volumes.size(0)

        optimizer.zero_grad()

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        Tuple[float, float]: (Average Loss, Average F0.5 Score)
    """
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    dataset_size = 0

    with torch.no_grad():
        for volumes, labels in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            batch_size = volumes.size(0)

            outputs = model(volumes)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size

            # Compute F0.5 score
            # Note: fbeta_score handles thresholding (default 0.5)
            score = fbeta_score(outputs, labels, beta=0.5)
            running_score += score * batch_size

            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_score = running_score / dataset_size

    return epoch_loss, epoch_score


def train_model(
    model, train_loader, val_loader, optimizer, device, num_epochs, patience, save_path
):
    """
    Orchestrates the training process with early stopping.

    Args:
        model (nn.Module): The neural network.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Torch optimizer.
        device (torch.device): Compute device.
        num_epochs (int): Maximum number of epochs.
        patience (int): Epochs to wait for improvement before early stopping.
        save_path (str or Path): Path to save the best model weights.
    """
    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = BCEDiceLoss()
    best_val_score = -1.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Phase
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5 Score: {val_score}")

        # Early Stopping Logic (Maximize F0.5 Score)
        if val_score > best_val_score:
            best_val_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Validation score improved. Model saved.")
        else:
            early_stop_counter += 1
            print(
                f"No improvement. EarlyStopping counter: {early_stop_counter}/{patience}"
            )

            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
