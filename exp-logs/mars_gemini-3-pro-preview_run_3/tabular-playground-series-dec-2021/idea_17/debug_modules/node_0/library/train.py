import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import ParallelDCNResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Executes the training pipeline with Early Stopping and Scheduler.

    Args:
        epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        early_stopping_patience (int): Patience for early stopping.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug (bool): If True, uses a subset of data for debugging.

    Returns:
        tuple: (best_model, test_loader, test_ids)
    """
    # Apply debug setting to Config so data_loader picks it up
    Config.DEBUG = debug

    seed_everything(Config.SEED)
    device = get_device()

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Determine input dimension from dataset
    input_dim = train_loader.dataset.X.shape[1]
    num_classes = Config.NUM_CLASSES

    print(f"Data Loaded. Input Dimension: {input_dim}, Num Classes: {num_classes}")

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=Config.DCN_RANK,
        resnet_hidden=Config.RESNET_HIDDEN_DIM,
        resnet_blocks=Config.RESNET_NUM_BLOCKS,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # Training Loop with Early Stopping
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision (no rounding)
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model to disk
            torch.save(best_model_wts, Config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, test_loader, test_ids


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions using the trained model and saves to submission.csv.
    """
    device = get_device()
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())

    # Map predictions back to original labels using Inverse Map
    final_preds = [Config.INVERSE_LABEL_MAP[p] for p in predictions]

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved.")


def train_and_predict():
    """
    Orchestrates the full pipeline: Training -> Inference -> Submission.
    """
    model, test_loader, test_ids = run_training()
    generate_submission(model, test_loader, test_ids)
