import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.models import CustomResNet, CustomDenseNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (N, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    epoch_loss = running_loss / count
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    # Handle edge case where batch might have only one class (unlikely in full val set but possible in debug)
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(architecture, seed, train_loader, val_loader):
    """
    Runs the full training loop for a specific architecture and seed.
    """
    # 1. Setup
    set_seed(seed)
    device = Config.DEVICE
    print(f"Starting training for Architecture: {architecture}, Seed: {seed}")

    # 2. Initialize Model
    if architecture == "resnet":
        model = CustomResNet(num_classes=Config.NUM_CLASSES)
    elif architecture == "densenet":
        model = CustomDenseNet(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)

    # 3. Optimizer & Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    save_path = Config.get_model_path(architecture, seed)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss}, "
            f"Val Loss: {val_loss}, "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Finished training {architecture} (seed {seed}). Best Val AUC: {best_auc}")
    return best_auc


def train_ensemble():
    """
    Main entry point to train all models in the ensemble.
    """
    # Ensure directories exist
    Config.setup_directories()

    # Load Data (cached)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    results = {}

    # Iterate over all combinations
    for arch in Config.ARCHITECTURES:
        for seed in Config.SEEDS:
            auc = run_training(arch, seed, train_loader, val_loader)
            results[f"{arch}_seed_{seed}"] = auc

    print("\nEnsemble Training Complete.")
    print("Final Results:")
    for k, v in results.items():
        print(f"{k}: AUC = {v}")
