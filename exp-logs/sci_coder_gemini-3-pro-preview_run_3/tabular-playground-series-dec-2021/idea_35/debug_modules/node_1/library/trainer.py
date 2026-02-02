import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import random
from library.config import Config


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility and configures CuDNN for performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Requirement: Disable strict CuDNN determinism to maximize kernel performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(model, train_loader, val_loader, epochs=None, device=None):
    """
    Manages the optimization loop, scheduler, and early stopping.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    if epochs is None:
        epochs = Config.EPOCHS

    if device is None:
        device = Config.DEVICE

    print(f"Starting training on {device} for {epochs} epochs...")

    # Optimizer: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau with aggressive decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stopping_patience = 10

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
        )

        # Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model to disk
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model
