import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, compute_auc


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Performs one epoch of training.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch in dataloader:
        # Move data to device
        numerical = batch["numerical"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model returns probabilities (sigmoid applied)
        preds = model(numerical, sequence)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get_avg("Loss")


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns: avg_loss, auc_score
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(numerical, sequence)

            # Calculate loss
            loss = criterion(preds, targets)

            # Update metrics
            metric_monitor.update("Loss", loss.item())

            # Store for AUC calculation
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute AUC
    auc = compute_auc(all_targets, all_preds)

    return metric_monitor.get_avg("Loss"), auc


def train_model(model, train_loader, val_loader, device):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Setup Output Directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Loss Function (Model output is Sigmoid, so we use BCELoss)
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Early Stopping Variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training on device: {device}")
    print(f"Epochs: {Config.EPOCHS} | Batch Size: {Config.BATCH_SIZE}")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best AUC! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
    return best_auc


def generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")

    # Load Best Model State
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()
    model.to(device)

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            ids = batch["id"]

            preds = model(numerical, sequence)

            all_ids.extend(ids.numpy())
            all_preds.extend(preds.cpu().numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Ensure ID is integer (it might be read as float sometimes)
    df_sub["id"] = df_sub["id"].astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {df_sub.shape}")
    print(f"Head:\n{df_sub.head()}")
