import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import random

import library.config as config
from library.data_utils import load_data
from library.dataset import ManufacturingDataset
from library.model import MRPFEModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    Computes the sum of losses across all 5 streams.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: returns list of outputs from 5 streams
        outputs = model(continuous, categorical)

        # Calculate combined loss (sum of BCE for each stream)
        loss = 0
        for output in outputs:
            loss += criterion(output, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * continuous.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model.
    Computes the ensemble prediction (mean of 5 streams) and calculates AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            outputs = model(continuous, categorical)

            # Calculate loss for monitoring
            loss = 0
            probs_sum = 0
            for output in outputs:
                loss += criterion(output, targets)
                probs_sum += torch.sigmoid(output)

            # Ensemble prediction: Arithmetic mean of probabilities
            avg_probs = probs_sum / len(outputs)

            running_loss += loss.item() * continuous.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(avg_probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_loss = running_loss / len(dataloader.dataset)

    # Handle edge case if only one class is present in batch (unlikely with stratification)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def run_training(nrows=None):
    """
    Main training loop with Early Stopping.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    print(f"Loading data (nrows={nrows})...")
    # load_data handles caching internally
    train_df, val_df, test_df, vocab_sizes = load_data(
        load_cached_data=True, nrows=nrows
    )

    # Create Datasets
    train_dataset = ManufacturingDataset(train_df, is_test=False)
    val_dataset = ManufacturingDataset(val_df, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    num_continuous = len(config.ALL_CONTINUOUS_FEATURES)
    model = MRPFEModel(vocab_sizes, num_continuous).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.MAX_LR, weight_decay=config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_model_path, vocab_sizes, num_continuous


def generate_submission(model_path, vocab_sizes, num_continuous, nrows=None):
    """
    Generates predictions for the test set using the best model and saves to CSV.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    print("Loading test data for submission...")
    # We only need test_df here, but load_data returns everything
    _, _, test_df, _ = load_data(load_cached_data=True, nrows=nrows)

    test_dataset = ManufacturingDataset(test_df, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model and Load Weights
    model = MRPFEModel(vocab_sizes, num_continuous).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)

            outputs = model(continuous, categorical)

            # Ensemble prediction
            probs_sum = 0
            for output in outputs:
                probs_sum += torch.sigmoid(output)

            avg_probs = probs_sum / len(outputs)
            all_preds.append(avg_probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Create Submission DataFrame
    # Using IDs from the dataset (preserved from original dataframe)
    submission_df = pd.DataFrame({"id": test_dataset.ids, "target": all_preds})

    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
