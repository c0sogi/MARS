import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_datasets
from library.model import BirdClassifier
from library.utils import seed_everything, save_state, calculate_metric


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    # Linear interpolation of targets for multi-label BCE
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        images, labels = mixup_data(images, labels, alpha, device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro-Averaged ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for metric calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc_score = calculate_metric(all_targets, all_preds)

    return epoch_loss, auc_score


def inference(model_path, test_loader, device):
    """
    Loads the best model and generates predictions for the test set.
    Saves the submission file.
    """
    # Load Model
    model = BirdClassifier(backbone=Config.BACKBONE, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []

    # Generate predictions
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    # Shape: (N_test, Num_Classes)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load Test Metadata to get Rec IDs
    # The dataset loader preserves the order of the CSV
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    rec_ids = df_test["rec_id"].values

    # Format Submission
    submission_rows = []
    for i, rec_id in enumerate(rec_ids):
        probs = all_preds[i]
        for species_id, prob in enumerate(probs):
            # ID format: rec_id * 100 + species_id
            row_id = int(rec_id * 100 + species_id)
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train():
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create Working Directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Data
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    print(
        f"Initializing model: {Config.BACKBONE} with GeM Pooling={Config.USE_GEM_POOLING}"
    )
    model = BirdClassifier(backbone=Config.BACKBONE, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss Function (BCEWithLogitsLoss includes Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 10
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MIXUP_ALPHA
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.9f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            save_state(model, best_model_path)
            print(f"New best model saved with AUC: {best_auc:.9f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

    total_time = time.time() - start_time
    print(
        f"Training complete in {total_time:.2f} seconds. Best Val AUC: {best_auc:.9f}"
    )

    # Inference
    print("Generating submission...")
    inference(best_model_path, test_loader, device)
