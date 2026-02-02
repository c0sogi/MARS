import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import from the provided library files
from library.config import Config
from library.dataset import CactusDataset
from library.model import HybridNarrowSEResNet
from library.utils import seed_everything, calculate_roc_auc


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape matches logits

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    auc_score = calculate_roc_auc(all_labels, all_probs)

    return epoch_loss, auc_score


def run_training(seed: int):
    """
    Runs the full training pipeline for a specific seed.

    Args:
        seed (int): The random seed to initialize the run.

    Returns:
        float: The best validation AUC achieved.
    """
    # 1. Reproducibility
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training for Seed {seed} on device: {device}")

    # 2. Data Preparation
    # Using metadata paths from Config
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, phase="train", load_cached_data=True
    )

    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH, phase="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HybridNarrowSEResNet()
    model = model.to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved for seed {seed} with AUC: {best_auc}")

    print(f"Training finished for Seed {seed}. Best Val AUC: {best_auc}")
    return best_auc
