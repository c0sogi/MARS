import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import transforms
from library.config import Config
from library.utils import set_seed, get_device, compute_roc_auc, print_metric
from library.data_loader import get_train_val_datasets, get_test_dataset
from library.model import MILEfficientNet


def get_augmentations():
    """
    Returns a Sequential module of transforms to be applied on the GPU.
    Includes Geometric (Flips) and Rotation.
    """
    return nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    )


def train_one_epoch(model, loader, optimizer, criterion, device, augmentor=None):
    """
    Trains the model for one epoch using Multi-Instance Learning (MIL).
    Strategy: Max-Pooling of logits (optimize the most confident candidate).
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for data, target, _ in loader:
        # data: (Batch, Candidates, Channels, H, W)
        # target: (Batch,)
        data, target = data.to(device), target.to(device)
        b, n, c, h, w = data.shape

        # Apply Augmentations
        if augmentor is not None:
            # Flatten candidates into batch dimension for augmentation
            # (B * N, C, H, W)
            flat_data = data.view(b * n, c, h, w)
            flat_data = augmentor(flat_data)
            # Reshape back
            data = flat_data.view(b, n, c, h, w)

        optimizer.zero_grad()

        # Forward Pass -> Logits: (Batch, Candidates)
        logits = model(data)

        # MIL Strategy: Max-Pooling
        # We take the maximum logit across candidates for each patient
        # This focuses the loss on the most "tumor-like" candidate
        max_logits, _ = torch.max(logits, dim=1)  # Shape: (Batch,)

        loss = criterion(max_logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * b

        # Collect metrics (Sigmoid for probability)
        probs = torch.sigmoid(max_logits).detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        all_targets.extend(targets)
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = compute_roc_auc(all_targets, all_probs)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Validates the model.
    Strategy: Max-Pooling of logits (consistent with training objective).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            b = data.size(0)

            # Forward -> Logits: (Batch, Candidates)
            logits = model(data)

            # Max-Pooling
            max_logits, _ = torch.max(logits, dim=1)

            loss = criterion(max_logits, target)
            running_loss += loss.item() * b

            probs = torch.sigmoid(max_logits).cpu().numpy()
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs)

    val_loss = running_loss / len(loader.dataset)
    val_auc = compute_roc_auc(all_targets, all_probs)

    return val_loss, val_auc


def run_training(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main execution function for the training pipeline.
    Handles setup, training loop, early stopping, and model saving.
    """
    set_seed()
    device = get_device()

    # 1. Load Data
    # Caching is handled internally by data_loader
    train_ds, val_ds = get_train_val_datasets(load_cached=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model, Optimizer, Loss
    model = MILEfficientNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Augmentation module on device
    augmentor = get_augmentations().to(device)

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, augmentor
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print_metric(f"Epoch {epoch}", "Train Loss", train_loss)
        print_metric(f"Epoch {epoch}", "Train AUC", train_auc)
        print_metric(f"Epoch {epoch}", "Val Loss", val_loss)
        print_metric(f"Epoch {epoch}", "Val AUC", val_auc)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Model saved at epoch {epoch}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break


def predict_and_submit(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves the submission file.
    Strategy: Mean-Pooling of probabilities (Ensemble-like robustness).
    """
    set_seed()
    device = get_device()

    # 1. Load Data
    test_ds = get_test_dataset(load_cached=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = MILEfficientNet().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No saved model found. Using random initialization.")

    model.eval()

    ids_list = []
    probs_list = []

    # 3. Inference Loop
    with torch.no_grad():
        for data, _, ids in test_loader:
            data = data.to(device)

            # Forward -> Logits: (Batch, Candidates)
            logits = model(data)

            # Inference Strategy: Mean-Pooling
            # We calculate probabilities for all candidates and average them.
            # This acts as Test Time Augmentation (TTA) via multiple views.
            probs = torch.sigmoid(logits)  # (Batch, Candidates)
            avg_probs = torch.mean(probs, dim=1)  # (Batch,)

            ids_list.extend(ids.numpy())
            probs_list.extend(avg_probs.cpu().numpy())

    # 4. Save Submission
    # Format IDs as 5-digit strings (e.g., 00001)
    formatted_ids = [f"{int(pid):05d}" for pid in ids_list]

    df = pd.DataFrame({"BraTS21ID": formatted_ids, "MGMT_value": probs_list})

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
