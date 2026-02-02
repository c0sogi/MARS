import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library import utils, dataset, model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    return running_loss / num_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    avg_loss = running_loss / num_samples

    # Concatenate all batches
    y_true = torch.cat(all_labels).numpy()
    y_scores = torch.cat(all_probs).numpy()

    # Calculate AUC
    auc_score = utils.calculate_roc_auc(y_true, y_scores)

    return avg_loss, auc_score


def run_training_seed(seed, debug_sample_size=None):
    """
    Runs the full training loop for a specific seed.
    Includes model initialization, optimizer setup, training loop,
    validation, early stopping, and model saving.
    """
    # 1. Reproducibility
    utils.seed_everything(seed)

    device = torch.device(Config.DEVICE)
    print(f"Starting training for Seed {seed} on device: {device}")

    # 2. Data Loading
    train_loader, val_loader, _ = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_sample_size=debug_sample_size,
    )

    # 3. Model Initialization
    net = model.CustomWideResNet()
    net = net.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )

    # 5. Training Loop with Early Stopping
    best_auc = -float("inf")
    patience_counter = 0
    best_model_path = Config.get_model_path(seed)

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(net.state_dict(), best_model_path)
            print(f"New best model saved for seed {seed} with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Finished training for Seed {seed}. Best AUC: {best_auc}")
