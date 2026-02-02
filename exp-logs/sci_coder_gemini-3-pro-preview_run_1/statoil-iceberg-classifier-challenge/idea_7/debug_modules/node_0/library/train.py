import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.model import IcebergResNet18
from library.data_loader import get_dataloaders, load_data, get_test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (Batch, 1)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def run_bag_training(bag_idx, train_idx, val_idx, device):
    """
    Trains a single model for a specific bag (bootstrap sample).
    """
    print(f"\n--- Starting training for Bag {bag_idx} ---")
    print(f"Train size: {len(train_idx)}, OOB Validation size: {len(val_idx)}")

    # Get DataLoaders for this split
    train_loader, val_loader = get_dataloaders(train_idx, val_idx)

    # Initialize Model
    model = IcebergResNet18().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_bag_{bag_idx}.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered for Bag {bag_idx} at epoch {epoch+1}")
                break

    print(f"Finished Bag {bag_idx}. Best OOB Val Loss: {best_val_loss:.10f}")
    return best_val_loss


def train_bagging_ensemble():
    """
    Main function to train the bagging ensemble.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data once to get indices and size
    # We use cached data if available
    _, _, train_labels, _, _, _ = load_data(load_cached_data=True)

    N = len(train_labels)
    indices = np.arange(N)

    bag_metrics = []

    for bag_idx in range(Config.NUM_BAGS):
        # Bootstrap Sampling: Sample N items with replacement
        train_idx = np.random.choice(indices, size=N, replace=True)

        # Out-of-Bag (OOB) samples for validation
        val_idx = np.setdiff1d(indices, train_idx)

        # Safety check for OOB existence
        if len(val_idx) == 0:
            print(f"Warning: Bag {bag_idx} has no OOB samples. Resampling...")
            while len(val_idx) == 0:
                train_idx = np.random.choice(indices, size=N, replace=True)
                val_idx = np.setdiff1d(indices, train_idx)

        val_loss = run_bag_training(bag_idx, train_idx, val_idx, device)
        bag_metrics.append(val_loss)

    print("\nEnsemble Training Complete.")
    print(
        f"Average OOB Loss across {Config.NUM_BAGS} bags: {np.mean(bag_metrics):.10f}"
    )


def predict_with_tta(model, images, angles, device):
    """
    Predicts with Test Time Augmentation: Original, H-Flip, V-Flip.
    Returns average probability.
    """
    model.eval()
    with torch.no_grad():
        # Move to device
        images = images.to(device)
        angles = angles.to(device)

        # 1. Original
        logits_orig = model(images, angles)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip (dim 3 is width)
        images_h = torch.flip(images, [3])
        logits_h = model(images_h, angles)
        probs_h = torch.sigmoid(logits_h)

        # 3. Vertical Flip (dim 2 is height)
        images_v = torch.flip(images, [2])
        logits_v = model(images_v, angles)
        probs_v = torch.sigmoid(logits_v)

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v) / 3.0

    return avg_probs.cpu().numpy()


def generate_submission():
    """
    Generates the submission file using the trained ensemble.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # Get Test Loader
    test_loader, test_ids = get_test_loader()

    # Initialize array to store predictions from all bags
    # Shape: (Num_Bags, Num_Test_Samples)
    all_bag_preds = []

    for bag_idx in range(Config.NUM_BAGS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_bag_{bag_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Model for Bag {bag_idx} not found at {model_path}. Skipping.")
            continue

        print(f"Predicting with Bag {bag_idx}...")

        # Load Model
        model = IcebergResNet18().to(device)
        model = load_checkpoint(model, model_path, device)

        bag_preds = []
        for images, angles in test_loader:
            probs = predict_with_tta(model, images, angles, device)
            bag_preds.append(probs)

        # Concatenate batches for this bag
        bag_preds = np.concatenate(bag_preds, axis=0).flatten()
        all_bag_preds.append(bag_preds)

    # Average across bags
    if not all_bag_preds:
        print("No predictions generated.")
        return

    all_bag_preds = np.array(all_bag_preds)
    avg_preds = np.mean(all_bag_preds, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())
