import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model_utils import ParallelDCNResNet

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model: PyTorch model.
        data_loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        avg_loss: Average loss over the dataset.
        accuracy: Accuracy over the dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    train_loader, val_loader, input_dim, num_classes=7, epochs=Config.EPOCHS
):
    """
    Executes the training pipeline with AdamW, ReduceLROnPlateau, and Early Stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        input_dim: Integer, number of input features.
        num_classes: Integer, number of target classes.
        epochs: Integer, maximum number of training epochs.

    Returns:
        model: The trained PyTorch model with best weights loaded.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.CrossEntropyLoss()

    # Early Stopping State
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation Phase ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss} Acc: {train_acc} | "
            f"Val Loss: {val_loss} Acc: {val_acc}"
        )

        # Update Scheduler
        scheduler.step(val_acc)

        # --- Early Stopping Check ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save(best_model_wts, Config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model


def predict_and_submit(model, test_loader, test_ids):
    """
    Generates predictions on the test set and saves the submission CSV.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        test_ids: Array of test IDs corresponding to the loader data.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    predictions = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Map 0-6 back to 1-7 (Original Class Labels)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
