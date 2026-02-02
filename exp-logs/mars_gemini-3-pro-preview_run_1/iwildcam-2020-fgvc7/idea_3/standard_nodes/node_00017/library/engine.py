import torch
import os
from library import config, utils


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network.
        loader (torch.utils.data.DataLoader): Training dataloader.
        criterion (torch.nn.Module): Loss function (e.g., FocalLoss).
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network.
        loader (torch.utils.data.DataLoader): Validation dataloader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs=config.EPOCHS,
    device=None,
    patience=config.EARLY_STOPPING_PATIENCE,
    save_path=config.BEST_MODEL_PATH,
):
    """
    Orchestrates the training process including logging, scheduling, and early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        criterion (Module): Loss function.
        num_epochs (int): Maximum number of epochs.
        device (torch.device): Device to use.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model checkpoint.
    """
    if device is None:
        device = utils.get_device()

    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"LR: {current_lr}")
        print(f"Train Loss: {train_loss} Acc: {train_acc}")
        print(f"Val Loss: {val_loss} Acc: {val_acc}")

        # Save best model and handle early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Ensure directory exists
            utils.ensure_directory(os.path.dirname(save_path))
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    return best_acc
