import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from library.configuration import Config
from library.utilities import set_seed, mixup_data, mixup_criterion, calculate_lrap
from library.network import ConvNeXtAudio
from library.data_loader import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, scheduler, device, config):
    """
    Training loop for one epoch.
    Applies Mixup augmentation and updates model weights.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        images, labels_a, labels_b, lam = mixup_data(
            images, labels, config.MIXUP_ALPHA, device
        )

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate Loss
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step Scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / num_batches


def valid_epoch(model, loader, criterion, device):
    """
    Validation loop for one epoch.
    Calculates Loss and LWLRAP score without augmentation.
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(loader)

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # Calculate Loss
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # Apply Sigmoid for metric calculation
            probs = torch.sigmoid(logits)

            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    # Calculate LWLRAP
    # Note: calculate_lrap expects numpy arrays
    lwlrap_score = calculate_lrap(all_targets, all_outputs)
    avg_loss = running_loss / num_batches

    return avg_loss, lwlrap_score


def run_training(config=Config, load_cached_data=True):
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        config, load_cached_data=load_cached_data
    )

    # 3. Model
    print("Initializing Model...")
    model = ConvNeXtAudio(config=config)
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        pct_start=config.PCT_START,
    )

    # 5. Training Loop
    best_score = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())

    # Early Stopping parameters
    patience = 7
    counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config
        )

        # Validate
        val_loss, val_score = valid_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val LWLRAP: {val_score}"
        )

        # Check for improvement (Maximize LWLRAP)
        if val_score > best_score:
            print(
                f"Validation score improved from {best_score} to {val_score}. Saving model..."
            )
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation LWLRAP: {best_score}")

    # 6. Save Best Model
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

    print(f"Saving best model to {config.BEST_MODEL_PATH}...")
    torch.save(best_model_wts, config.BEST_MODEL_PATH)

    return best_score
