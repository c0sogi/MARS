import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.data_loader import get_dataloaders
from library.model import GatedTransformerResFunnelHybrid


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Move inputs to device
        continuous = batch["continuous"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(continuous, categorical)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        tuple: (Average Loss, AUC Score)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = model(continuous, categorical)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / total_samples

    # Concatenate all batches for metric calculation
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    auc_score = calculate_auc(y_true, y_pred)

    return avg_loss, auc_score


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)

            logits = model(continuous, categorical)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    # Flatten predictions
    y_pred = np.vstack(all_preds).flatten()

    # Load Test Metadata to ensure correct ID mapping
    # The test_loader is sequential and matches the order of test_metadata
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    if len(y_pred) != len(test_meta):
        print(
            f"Warning: Number of predictions ({len(y_pred)}) does not match metadata ({len(test_meta)})."
        )

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_meta["id"], "target": y_pred})

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for AdamW.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.setup()  # Ensure working directory exists
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Prepare Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Initialize Model
    model = GatedTransformerResFunnelHybrid()
    model.to(device)

    # 4. Optimizer & Scheduler
    # Using AdamW with high weight decay as specified
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function (combines Sigmoid + BCELoss)
    criterion = nn.BCEWithLogitsLoss()

    # Aggressive Step Decay Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training Loop
    best_auc = 0.0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Update Learning Rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - start_time

        # Log Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.5e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"New best AUC achieved: {best_auc:.10f}. Saving checkpoint...")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_auc, Config.MODEL_SAVE_PATH
            )

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # 6. Inference & Submission
    print("Loading best model for inference...")
    # Load the best state dict into the model
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=Config.DEVICE)
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} with AUC {checkpoint['score']:.10f}"
    )

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
