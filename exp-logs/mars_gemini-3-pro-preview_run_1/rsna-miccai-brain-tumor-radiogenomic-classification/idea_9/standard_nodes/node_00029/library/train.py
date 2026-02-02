import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import MGMTClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (Batch, 1) for BCE

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count if count > 0 else 0.0


def validate_epoch(model, loader, device):
    """
    Evaluates the model on the validation set using Subject-Level Aggregation.
    Returns the ROC AUC score.
    """
    model.eval()
    preds = []

    # Iterate over slices
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs.flatten())

    # Access the underlying dataframe to map slices back to subjects
    # The loader is not shuffled, so order matches the dataset.df
    df = loader.dataset.df.copy()
    df["pred"] = preds

    # Consensus Aggregation: Mean prediction per subject
    subject_scores = df.groupby("BraTS21ID").agg(
        {"pred": "mean", "MGMT_value": "first"}
    )

    y_true = subject_scores["MGMT_value"].values
    y_pred = subject_scores["pred"].values

    # Calculate AUC
    try:
        # Check if we have both classes to avoid ValueError
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    return auc


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    preds = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs.flatten())

    # Map slices to subjects
    df = loader.dataset.df.copy()
    df["MGMT_value"] = preds  # Reuse column name for prediction

    # Aggregate
    submission_df = df.groupby("BraTS21ID")["MGMT_value"].mean().reset_index()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug_sample_size=None, epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        debug_sample_size=debug_sample_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = MGMTClassifier().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = validate_epoch(model, val_loader, device)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # 6. Generate Submission with Best Model
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for submission...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    return best_model_path
