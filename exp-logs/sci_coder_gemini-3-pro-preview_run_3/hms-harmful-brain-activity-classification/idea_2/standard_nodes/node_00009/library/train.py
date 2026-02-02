import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from library import config, utils, data, model


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, scheduler=None
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training(debug=False):
    """
    Main function to run the training pipeline.
    """
    utils.seed_everything(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_batch_size=config.BATCH_SIZE,
        val_batch_size=config.BATCH_SIZE,
        debug=debug,
    )

    # Model Initialization
    net = model.EEGWaveNet(pretrained=True)
    net.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    # T_max is the number of steps if stepping per batch, or epochs if stepping per epoch.
    # Here we step per epoch for simplicity in the loop below.
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Loss Function
    criterion = utils.KL_Loss()

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Training Loop Variables
    best_loss = float("inf")
    best_model_wts = None
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            net, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss = validate(net, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = net.state_dict()
            torch.save(best_model_wts, config.MODEL_PATH)
            print(f"  >>> Model Saved (Improved Loss: {best_loss})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  >>> No improvement. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_loss}")

    # Run Inference
    predict(test_loader, device, debug=debug)


def predict(test_loader, device, debug=False):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Starting inference on test set...")

    # Load Best Model
    net = model.EEGWaveNet(pretrained=False)
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()

    probs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = net(inputs)

            # Apply Softmax to get probabilities (KL Loss uses LogSoftmax, but submission needs Probs)
            batch_probs = torch.softmax(logits, dim=1)

            probs.append(batch_probs.cpu().numpy())

    # Concatenate all batches
    probs = np.concatenate(probs)

    # Load Test Metadata to get EEG IDs
    test_df = pd.read_csv(config.TEST_CSV)
    if debug:
        test_df = test_df.head(config.DEBUG_SIZE)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "eeg_id": test_df["eeg_id"],
            "seizure_vote": probs[:, 0],
            "lpd_vote": probs[:, 1],
            "gpd_vote": probs[:, 2],
            "lrda_vote": probs[:, 3],
            "grda_vote": probs[:, 4],
            "other_vote": probs[:, 5],
        }
    )

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())
