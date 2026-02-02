import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as F

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_DIR,
    METADATA_DIR,
)
from library.data_loader import get_dataloader
from library.model_factory import AsymmetricEfficientNet
from library.utils import set_seed

# Ensure reproducibility
set_seed()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * inputs.size(0)

        # Collect predictions for AUC
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)

    # Handle edge case where batch might contain only one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Main driver for the training pipeline.
    """
    print(f"Initializing training on device: {DEVICE}")

    # 1. Load Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # 2. DataLoaders
    # get_dataloader handles anchor caching internally
    train_loader = get_dataloader(
        df_train, BATCH_SIZE, phase="train", load_cached_anchors=True
    )
    val_loader = get_dataloader(
        df_val, BATCH_SIZE, phase="val", load_cached_anchors=True
    )

    # 3. Model Setup
    model = AsymmetricEfficientNet().to(DEVICE)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 5. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} | Val AUC: {val_auc}")

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def predict_and_submit(model_path):
    """
    Runs inference on the test set using Test-Time Augmentation (TTA)
    and generates the submission file.
    """
    print("Starting inference...")

    # 1. Load Test Data
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_csv_path)

    # Note: Phase 'test' disables random augmentations in the dataset
    test_loader = get_dataloader(
        df_test, BATCH_SIZE, phase="test", load_cached_anchors=True
    )

    # 2. Load Model
    model = AsymmetricEfficientNet().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    model.eval()

    predictions = []

    # 3. Inference Loop with TTA
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)

            # TTA Strategy:
            # 1. Original
            out_orig = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip
            inputs_h = F.hflip(inputs)
            out_h = torch.sigmoid(model(inputs_h))

            # 3. Vertical Flip
            inputs_v = F.vflip(inputs)
            out_v = torch.sigmoid(model(inputs_v))

            # Average predictions
            avg_preds = (out_orig + out_h + out_v) / 3.0

            predictions.extend(avg_preds.cpu().numpy().flatten())

    # 4. Save Submission
    submission_df = df_test[["BraTS21ID"]].copy()
    submission_df["MGMT_value"] = predictions

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
