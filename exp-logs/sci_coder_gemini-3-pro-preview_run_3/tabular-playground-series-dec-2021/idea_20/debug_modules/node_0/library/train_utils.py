import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

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


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Manages the full training lifecycle including optimization, scheduling,
    and early stopping with deepcopy checkpointing.
    """
    # Ensure working directory exists for saving the best model
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler: Decays LR to 0 over the total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model to disk immediately
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(best_model_wts, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc}")

    # Load best weights into the model before returning
    model.load_state_dict(best_model_wts)
    return model
