import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SEED,
    INPUT_DROPOUT_PROB,
    SAMPLE_SUBMISSION_PATH,
)
from library.utils import get_device, seed_everything
from library.model import RNVSNetwork
from library.dataset import get_dataloaders


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in batch (though unlikely in full val set)
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return total_loss, auc_score


def predict_and_submit(model, test_loader, device, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for inputs, subject_ids in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            ids_list.extend(subject_ids.numpy())
            preds_list.extend(probs.cpu().numpy().flatten())

    # Create DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": preds_list})

    # Ensure IDs are formatted correctly (though sample submission uses int)
    # The sample submission provided in description shows BraTS21ID as int.

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False, epochs=EPOCHS):
    """
    Main driver function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model
    print("Initializing RN-VS Network (Standard 3-Channel)...")
    model = RNVSNetwork().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    patience = 5
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.10f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    predict_and_submit(model, test_loader, device)

    return best_auc
