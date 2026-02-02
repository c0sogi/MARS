import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config, TrainConfig
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import ParallelLowRankDCNResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
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


def validate(model, loader, criterion, device):
    """
    Performs validation loop.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Cosine Annealing.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Initialize Model
    model = ParallelLowRankDCNResNet().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=TrainConfig.LR, weight_decay=TrainConfig.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TrainConfig.EPOCHS, eta_min=0
    )

    # Early Stopping Variables
    best_model_state = None
    best_val_acc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, TrainConfig.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        if epoch % TrainConfig.VAL_CHECK_INTERVAL == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Update Scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch}/{TrainConfig.EPOCHS} - "
                f"LR: {current_lr} - "
                f"Train Loss: {train_loss} - Train Acc: {train_acc} - "
                f"Val Loss: {val_loss} - Val Acc: {val_acc}"
            )

            # Early Stopping Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Deepcopy to preserve exact weights
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                # Save checkpoint
                torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1

            if patience_counter >= TrainConfig.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc}"
                )
                break
        else:
            scheduler.step()
            print(
                f"Epoch {epoch}/{TrainConfig.EPOCHS} - Train Loss: {train_loss} - Train Acc: {train_acc}"
            )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training complete. Loaded best model with Val Acc: {best_val_acc}")
    else:
        print("Training complete (no improvement found).")

    return model


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = get_device()
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    # Convert 0-indexed predictions (0-6) to 1-indexed targets (1-7)
    predictions = np.array(predictions) + 1

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Run Training
    model = run_training(train_loader, val_loader)

    # Generate Submission
    generate_submission(model, test_loader, test_ids)
