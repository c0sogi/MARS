import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_lwlrap, save_checkpoint, AverageMeter
from library.dataset import AudioDataset, mixup_data
from library.model import AudioClassifier


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        # Apply Mixup
        data, y_a, y_b, lam = mixup_data(
            data, target, alpha=Config.mixup_alpha, device=device
        )

        optimizer.zero_grad()
        output = model(data)

        # Mixup loss calculation
        loss = criterion(output, y_a) * lam + criterion(output, y_b) * (1.0 - lam)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate LWLRAP
    lwlrap = calculate_lwlrap(all_targets, all_preds)

    return losses.avg, lwlrap


def run_training():
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    set_seed(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # Initialize Datasets and Loaders
    train_dataset = AudioDataset(Config.train_csv_path, mode="train")
    val_dataset = AudioDataset(Config.val_csv_path, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = AudioClassifier()
    model = model.to(device)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Training State
    best_lwlrap = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_lwlrap = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch}/{Config.epochs} - Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val LWLRAP: {val_lwlrap}")

        # Checkpointing and Early Stopping
        if val_lwlrap > best_lwlrap:
            best_lwlrap = val_lwlrap
            save_checkpoint(model, Config.model_save_path)
            print(f"New best model saved to {Config.model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best LWLRAP: {best_lwlrap}")


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    set_seed(Config.seed)
    device = Config.device

    if not os.path.exists(Config.model_save_path):
        print(f"Error: Model checkpoint not found at {Config.model_save_path}")
        return

    print("Loading model for inference...")
    model = AudioClassifier()
    state_dict = torch.load(Config.model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Test Dataset and Loader
    test_dataset = AudioDataset(Config.test_csv_path, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Construct Submission DataFrame
    fnames = test_dataset.df["fname"].values
    classes = test_dataset.classes

    submission_df = pd.DataFrame(all_preds, columns=classes)
    submission_df.insert(0, "fname", fnames)

    # Save to CSV
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
