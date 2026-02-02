import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

from library.config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_WORKERS,
    HIDDEN_DIM,
    NUM_CROSS_LAYERS,
    DROPOUT_RATE,
    NUM_CLASSES,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
)
from library.utils import seed_everything, ModelCheckpoint
from library.model import ParallelDCNResNet
from library.data_loader import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
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
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
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
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def predict_and_submit(model, test_loader, test_ids, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating predictions...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            # inputs is a list/tuple from TensorDataset, take first element
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Map 0-6 back to 1-7
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission dataframe
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    load_cached_data=True,
):
    """
    Orchestrates the full training lifecycle.
    """
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_cross_layers=NUM_CROSS_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )

    criterion = nn.CrossEntropyLoss()

    # 5. Checkpointing
    checkpoint = ModelCheckpoint(mode="max")

    # 6. Training Loop
    print("Starting training...")
    patience_counter = 0
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Checkpoint & Early Stopping
        improved = checkpoint.step(val_acc, model)
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint.save_best(MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc}")

    # 7. Prediction
    # Load best model weights
    model = checkpoint.load_best(model)
    predict_and_submit(model, test_loader, test_ids, device, SUBMISSION_PATH)
