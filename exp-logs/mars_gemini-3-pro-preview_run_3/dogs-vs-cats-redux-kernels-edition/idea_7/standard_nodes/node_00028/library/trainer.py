import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_device
from library.model_factory import create_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape [Batch, 1]

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size
    return val_loss


def train_model(model_name: str, train_loader, val_loader, patience: int = 3):
    """
    Orchestrates the training process for a specific model architecture.

    Args:
        model_name (str): The name of the backbone to train (e.g., 'resnet50.a1_in1k').
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        str: Path to the saved best model checkpoint.
    """
    device = get_device()
    print(f"Initializing model: {model_name} on {device}")

    # Create model
    model = create_model(model_name, pretrained=True, num_classes=1)
    model = model.to(device)

    # Loss Function: BCEWithLogitsLoss strictly without label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Training Loop Variables
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(
                f"No improvement in validation loss. Patience: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete for {model_name}. Best Val Loss: {best_val_loss}")
    return best_model_path
