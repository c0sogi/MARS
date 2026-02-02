import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect for metrics
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
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
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    epochs=15,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-2,
    patience=5,
    save_dir="./working/idea_13",
):
    """
    Executes the training pipeline with Early Stopping.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    # Get DataLoaders
    # Using load_cached_data=True as per requirement to use caching if available
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Initialize Model
    model = AsymmetricEfficientNet(pretrained=True)
    model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Train AUC: {train_auc} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    return best_auc


def predict_and_submit(
    model_path="./working/idea_13/best_model.pth",
    output_file="./submission/submission.csv",
    batch_size=32,
):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA: Average of Original, Horizontal Flip, and Vertical Flip.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get Test Loader
    _, _, test_loader = get_dataloaders(batch_size=batch_size, load_cached_data=True)

    # Load Model
    model = AsymmetricEfficientNet(pretrained=False)  # Weights loaded from file
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: Model path {model_path} not found. Predictions will be random."
        )

    model.to(device)
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for inputs, sids in test_loader:
            inputs = inputs.to(device)

            # TTA 1: Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA 2: Horizontal Flip (dim 3: W)
            inputs_h = torch.flip(inputs, dims=[3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # TTA 3: Vertical Flip (dim 2: H)
            inputs_v = torch.flip(inputs, dims=[2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average Probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            predictions.extend(avg_probs.cpu().numpy().flatten())
            # Cite debug_lesson_5: Cast NumPy scalars to native Python ints
            ids.extend([int(x) for x in sids.numpy()])

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Format BraTS21ID as 5-digit string
    df_sub["BraTS21ID"] = df_sub["BraTS21ID"].astype(int).apply(lambda x: f"{x:05d}")

    df_sub.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
