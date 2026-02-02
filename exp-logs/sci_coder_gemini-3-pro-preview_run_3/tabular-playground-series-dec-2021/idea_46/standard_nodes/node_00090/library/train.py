import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import get_dataloaders
from library.model import AsymmetricParallelNet, predict


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
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

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def run_training(
    model, train_loader, val_loader, device, epochs=60, lr=1e-3, patience=15
):
    """
    Manages the full training loop with scheduler and early stopping.
    """
    criterion = nn.CrossEntropyLoss()
    # AdamW with Decoupled Weight Decay as per strategy
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # ReduceLROnPlateau with aggressive decay factor 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, verbose=True
    )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Scheduler step based on validation accuracy
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def main(epochs=60, batch_size=4096, patience=15, lr=1e-3):
    """
    Main execution function to load data, train, and generate submission.
    """
    # 1. Setup
    seed_everything(42, deterministic=False)
    device = get_device()

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=True, num_workers=4
    )

    # 3. Model Initialization
    # Get input dimension from a sample batch
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]

    # Classes 0-6 (mapped from 1-7)
    num_classes = 7

    print(f"Initializing model with Input Dim: {input_dim}, Classes: {num_classes}")
    model = AsymmetricParallelNet(input_dim, num_classes).to(device)

    # 4. Training
    model = run_training(
        model, train_loader, val_loader, device, epochs=epochs, lr=lr, patience=patience
    )

    # 5. Prediction
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # 6. Submission
    save_submission(predictions, test_ids, "./submission/submission.csv")
