import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_score
from library.dataset import get_dataloaders
from library.model import MultiScaleResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # Squeeze outputs to match label shape (B,)
        loss = criterion(outputs.view(-1), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs).view(-1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    val_auc = calculate_score(all_targets, all_preds)

    return val_loss, val_auc


def train_model(seed):
    """
    Runs the training pipeline for a specific seed.
    Initializes model, optimizer, scheduler, and runs the training loop.
    Saves the best model based on Validation AUC.

    Args:
        seed (int): The random seed for this run.

    Returns:
        float: The best validation AUC achieved.
    """
    seed_everything(seed)
    device = torch.device(Config.DEVICE)

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = MultiScaleResNet()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    print(f"Starting training for Seed {seed} on {device}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training finished for Seed {seed}. Best Val AUC: {best_val_auc:.10f}")
    return best_val_auc


def generate_submission():
    """
    Generates submission file using Test Time Augmentation (TTA) and Ensemble Averaging.
    Loads models for all seeds defined in Config, predicts on test set, averages results,
    and saves to submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # Load Test Loader
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Collect predictions from all seeds
    ensemble_preds = []

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for seed {seed}...")
        model = MultiScaleResNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        seed_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # TTA 1: Original
                out1 = torch.sigmoid(model(images)).view(-1)

                # TTA 2: Horizontal Flip
                out2 = torch.sigmoid(model(torch.flip(images, [3]))).view(-1)

                # TTA 3: Vertical Flip
                out3 = torch.sigmoid(model(torch.flip(images, [2]))).view(-1)

                # Average TTA
                avg_out = (out1 + out2 + out3) / 3.0

                seed_preds.append(avg_out.cpu().numpy())

        ensemble_preds.append(np.concatenate(seed_preds))

    if not ensemble_preds:
        print("Error: No predictions generated.")
        return

    # Average across seeds (Ensemble averaging)
    final_preds = np.mean(ensemble_preds, axis=0)

    # Prepare submission dataframe
    # The dataset class stores IDs in the same order as the loader iterates
    test_ids = test_loader.dataset.ids

    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
