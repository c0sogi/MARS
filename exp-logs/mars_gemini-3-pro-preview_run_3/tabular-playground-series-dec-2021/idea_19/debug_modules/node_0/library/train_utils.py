import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.model_utils import ParallelDCNResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training: Forward pass, Loss, Backward pass, Optimizer step.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(train_loader, val_loader, input_dim, num_classes):
    """
    Orchestrates the full training lifecycle:
    - Model/Optimizer/Scheduler Initialization
    - Epoch Loop
    - Validation & Logging
    - Early Stopping (via deepcopy)
    """
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Initialize Model using the imported class
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        low_rank_factor=Config.LOW_RANK_FACTOR,
        num_cross_layers=3,
        num_res_blocks=3,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Checkpointing logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print(
        f"Training complete in {total_time}s. Best Validation Accuracy: {best_val_acc}"
    )

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_predictions(model, test_loader, test_ids, submission_path):
    """
    Runs inference on the test set and saves predictions to a CSV file.
    Maps 0-based model outputs back to 1-based class labels.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            # Map 0-6 back to 1-7
            predicted_classes = predicted.cpu().numpy() + 1
            predictions.extend(predicted_classes)

    predictions = np.array(predictions)

    if len(predictions) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of IDs ({len(test_ids)})."
        )

    # Create submission DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
