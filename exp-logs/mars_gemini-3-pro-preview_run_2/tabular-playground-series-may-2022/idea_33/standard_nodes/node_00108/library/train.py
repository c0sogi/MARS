import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import library.config as config
import library.utils as utils
import library.data as data
import library.model as model


def train_one_epoch(model_net, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    Handles Multi-Sample Dropout by averaging loss across all heads.
    """
    model_net.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move data to device
        cont = batch["continuous"].to(device, non_blocking=True)
        cat = batch["categorical"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, 5) due to Multi-Sample Dropout
        outputs = model_net(cont, cat)

        # Compute loss
        # We calculate BCE loss for each head and average them
        loss = 0.0
        for i in range(config.MSD_NUM_HEADS):
            loss += criterion(outputs[:, i], targets)

        loss = loss / config.MSD_NUM_HEADS

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model_net, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes AUC based on the average probability of the MSD heads.
    """
    model_net.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass -> (Batch, 5)
            logits = model_net(cont, cat)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Average probabilities across heads for final prediction
            avg_probs = probs.mean(dim=1)

            all_targets.append(targets.cpu())
            all_preds.append(avg_probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc = utils.compute_auc(all_targets, all_preds)
    return auc


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    load_cached_data=True,
):
    """
    Main training orchestration function.
    """
    # 1. Setup
    utils.seed_everything(config.RANDOM_STATE)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = data.get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print("Initializing Model...")
    net = model.ManufacturingNet()
    net.to(device)

    # 4. Optimizer & Scheduler
    # Use utility to separate params for weight decay
    optimizer_grouped_parameters = utils.get_optimizer_params(
        net, weight_decay=config.WEIGHT_DECAY_PARAMS
    )

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY_PARAMS,  # Default fallback, though groups override
    )

    scheduler = StepLR(
        optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)

        # Validate
        val_auc = evaluate(net, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"New best model found! Saving to {config.MODEL_SAVE_PATH}")
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(batch_size=config.BATCH_SIZE, load_cached_data=True):
    """
    Loads the best model and generates predictions for the test set.
    """
    device = torch.device(config.DEVICE)

    # Load Data
    _, _, test_loader = data.get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Load Model
    print(f"Loading best model from {config.MODEL_SAVE_PATH}...")
    net = model.ManufacturingNet()
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.to(device)
    net.eval()

    ids_list = []
    preds_list = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            ids = batch["id"].numpy()

            # Forward pass
            logits = net(cont, cat)

            # Average probabilities across 5 heads
            probs = torch.sigmoid(logits).mean(dim=1)

            ids_list.extend(ids)
            preds_list.extend(probs.cpu().numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids_list, "target": preds_list})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(df_sub.head())
