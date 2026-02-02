import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (str): The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        cont = batch["continuous"].to(device)
        cat = batch["categorical"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(cont, cat)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (str): The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            outputs = model(cont, cat)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Store predictions and targets for AUC calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    auc_score = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc_score


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    criterion=None,
    device=Config.DEVICE,
    epochs=Config.EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the training process with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer, optional): Optimizer instance. Defaults to AdamW from Config.
        criterion (Module, optional): Loss function. Defaults to BCELoss.
        device (str): Device to train on.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        nn.Module: The trained model with the best weights loaded.
    """
    model.to(device)

    # Default Optimizer if not provided
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    # Default Criterion if not provided
    if criterion is None:
        criterion = nn.BCELoss()

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": best_model_state,
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best AUC: {best_auc}"
                )
                break

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights.")

    return model
