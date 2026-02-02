import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything
from library.model import GNHRNet
from library.data_loader import get_dataloaders

# Constants
CHECKPOINT_DIR = "./working/idea_34/"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate ROC AUC
    # Handle edge case where only one class is present in the validation set
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    epochs=20,
    batch_size=8,
    lr=1e-4,
    patience=5,
    num_workers=4,
    load_cached_data=True,
    limit_data=None,
    seed=42,
):
    """
    Main function to train the model with Early Stopping.
    """
    seed_everything(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Get DataLoaders
    dataloaders = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
        limit_data=limit_data,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Initialize Model
    # in_chans=64 corresponds to 16 slices * 4 modalities
    model = GNHRNet(
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop variables
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_auc


def predict_and_submit(
    batch_size=8,
    num_workers=4,
    load_cached_data=True,
    output_dir=SUBMISSION_DIR,
):
    """
    Loads the best model, generates predictions on the test set, and saves submission.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get Test DataLoader
    dataloaders = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )
    test_loader = dataloaders["test"]

    # Initialize Model
    model = GNHRNet(
        model_name="efficientnet_b0", pretrained=False, in_chans=64, num_classes=1
    )

    # Load Weights
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded model weights from {BEST_MODEL_PATH}")
    else:
        print("Warning: Best model weights not found! Using random initialization.")

    model = model.to(device)
    model.eval()

    # Inference
    predictions = []
    ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs, _, batch_ids in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions.extend(probs)
            ids.extend(batch_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Save to CSV
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
