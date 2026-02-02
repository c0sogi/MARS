import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    auc_score = utils.calculate_roc_auc(all_labels, all_probs)

    return avg_loss, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    save_path,
    epochs=config.EPOCHS,
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    patience=config.PATIENCE,
):
    """
    Orchestrates the training process with optimization, scheduling, and early stopping.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to use.
        save_path (str): Path to save the best model weights.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.

    Returns:
        float: Best validation AUC score achieved.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"Validation AUC improved. Model saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in AUC. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
