import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.data import get_data_loaders
from library.model import AsymmetricDCNResNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
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
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Executes validation step.
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
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def predict(model, loader, device, debug_sample_size=None):
    """
    Generates predictions for the test set and saves to submission file.
    """
    model.eval()
    predictions = []

    # Generate predictions
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Map 0-6 back to 1-7 (Dataset classes are 1-based)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Load IDs
    # We must reload the test dataframe to get IDs as they are not in the loader
    # The loader only yields X tensors.
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_DATA_PATH}")

    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    if debug_sample_size is not None:
        df_test = df_test.iloc[:debug_sample_size]

    ids = df_test[Config.ID_COL].values

    # Ensure lengths match
    if len(ids) != len(predictions):
        print(f"Warning: Length mismatch. IDs: {len(ids)}, Preds: {len(predictions)}")
        min_len = min(len(ids), len(predictions))
        ids = ids[:min_len]
        predictions = predictions[:min_len]

    # Create submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    create_submission=True,
):
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        load_cached_data=True,
        debug_sample_size=debug_sample_size,
    )

    # Determine input dimension from dataset
    # train_loader.dataset is a ForestCoverDataset, .X is a tensor
    input_dim = train_loader.dataset.X.shape[1]
    print(f"Input Dimension: {input_dim}")

    # 2. Model Initialization
    model = AsymmetricDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_layers=Config.DCN_LAYERS,
        resnet_blocks=Config.RESNET_BLOCKS,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Time: {elapsed:.2f}s - "
            f"LR: {current_lr:.2e} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Train Acc: {train_acc:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"Val Acc: {val_acc}"
        )

        # Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save Checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }
            save_checkpoint(checkpoint, Config.MODEL_SAVE_PATH)
            print(f"    New best model saved! Accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(
                f"    No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # 5. Prediction / Submission
    if create_submission:
        print("Generating submission...")
        predict(model, test_loader, device, debug_sample_size)

    return model
