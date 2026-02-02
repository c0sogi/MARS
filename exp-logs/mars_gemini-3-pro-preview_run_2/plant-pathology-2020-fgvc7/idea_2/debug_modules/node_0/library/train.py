import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_loaders, load_dataset_df
from library.model import AppleEfficientNet


def calculate_class_weights(device, load_cached_data=True):
    """
    Calculates class weights based on the inverse frequency of classes in the training set.
    """
    # Load the training dataframe using the same cache mechanism as data.py
    df = load_dataset_df(
        Config.TRAIN_METADATA_PATH, "train_cache.parquet", load_cached_data
    )

    # Calculate counts for each class
    # Config.CLASS_LABELS defines the order: ["healthy", "multiple_diseases", "rust", "scab"]
    counts = df[Config.CLASS_LABELS].sum().values

    # Calculate weights: Total / (Num_Classes * Count)
    # This balances the contribution of each class to the loss
    total_samples = counts.sum()
    num_classes = len(counts)
    weights = total_samples / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(epoch, model, loss_fn, optimizer, loader, device, scheduler=None):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        # CrossEntropyLoss supports soft targets (probabilities)
        loss = loss_fn(logits, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def valid_one_epoch(epoch, model, loss_fn, loader, device):
    """
    Validation loop for one epoch.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            running_loss += loss.item()

            # Apply softmax to get probabilities for AUC calculation
            preds = torch.softmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate Metric (Mean Column-wise ROC AUC)
    auc_score = calculate_metric(all_labels, all_preds)

    return avg_loss, auc_score


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loaders
    train_loader, val_loader = get_loaders(load_cached_data=load_cached_data)

    # 3. Model
    model = AppleEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 4. Loss Function (Weighted Cross Entropy with Label Smoothing)
    class_weights = calculate_class_weights(device, load_cached_data=load_cached_data)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 6. Training Loop
    best_auc = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Epochs: {Config.EPOCHS}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler
        )

        # Validate
        val_loss, val_auc = valid_one_epoch(epoch, model, loss_fn, val_loader, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
