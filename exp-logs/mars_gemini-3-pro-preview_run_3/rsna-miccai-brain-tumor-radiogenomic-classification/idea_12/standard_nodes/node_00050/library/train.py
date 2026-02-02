import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE,
    DEVICE,
    WORKING_DIR,
)
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import SliceGroupedFusionNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        # Ensure targets are (B, 1) float tensor for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC (handle single-class edge case)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, loader, criterion, device):
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

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []
    ids_list = []

    print(f"Generating predictions for {len(loader.dataset)} test samples...")

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids_list.extend(ids)

    # Create submission DataFrame
    df = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": predictions})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Main pipeline: Setup -> Train -> Evaluate -> Save Best -> Predict.
    """
    # 1. Setup
    seed_everything()
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_dataloaders handles the caching logic internally via generate_dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = SliceGroupedFusionNet()
    model.to(device)

    # 4. Optimization
    # Explicitly using 0.0 weight decay as per instructions
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop with Early Stopping
    best_val_auc = -1.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} | Val AUC: {val_auc}")

        # Save Best Model
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved ({best_val_auc} -> {val_auc}). Saving model to {MODEL_SAVE_PATH}..."
            )
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("-" * 30)

    print(f"Training complete. Best Validation AUC: {best_val_auc}")

    # 6. Submission Generation
    # Load the best model weights
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading best model for submission...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model weights.")

    generate_submission(model, test_loader, device, SUBMISSION_FILE)
