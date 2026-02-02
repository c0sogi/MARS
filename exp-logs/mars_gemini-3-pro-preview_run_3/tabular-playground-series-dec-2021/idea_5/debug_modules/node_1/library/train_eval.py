import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import build_model
from library.data_utils import get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def train_model(config=Config, load_cached_data=True):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Ensure reproducibility
    config.setup()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=load_cached_data
    )

    # Determine input dimension from dataset
    # train_loader.dataset is a TensorDataset, tensors[0] is X
    input_dim = train_loader.dataset.tensors[0].shape[1]
    print(f"Input Feature Dimension: {input_dim}")

    # Build Model
    model = build_model(input_dim, config).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    # Training Loop Variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")
    epochs_no_improve = 0

    start_time = time.time()

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f}")
        print(f"Val Loss:   {val_loss:.10f} | Val Acc:   {val_acc:.10f}")

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping & Model Checkpointing
        # We prioritize Accuracy, then Loss
        improved = False
        if val_acc > best_acc:
            improved = True
        elif val_acc == best_acc and val_loss < best_loss:
            improved = True

        if improved:
            best_acc = val_acc
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best model found! (Acc: {best_acc:.10f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.10f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save model to disk
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Generate Predictions
    predict(model, test_loader, config)

    return model


def predict(model, test_loader, config=Config):
    """
    Generates predictions for the test set and saves submission file.
    """
    print("Generating predictions on test set...")
    device = torch.device(config.DEVICE)
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            # inputs is a list containing the tensor, or just the tensor?
            # TensorDataset with 1 tensor returns a tuple (tensor,)
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0]

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Load Test IDs
    # We rely on the cache path defined in Config, assuming data_utils has run
    if not os.path.exists(config.TEST_IDS_PATH):
        raise FileNotFoundError(
            f"Test IDs not found at {config.TEST_IDS_PATH}. Run training/data loading first."
        )

    test_ids = np.load(config.TEST_IDS_PATH)

    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )

    # Convert 0-indexed predictions back to 1-indexed labels
    final_preds = np.array(predictions) + 1

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {config.ID_COL: test_ids, config.TARGET_COL: final_preds}
    )

    # Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())
