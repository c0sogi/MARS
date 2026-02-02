import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.utils import set_seed
from library.data import get_dataloaders
from library.model import RFMHDNetwork


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

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

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate results
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case where only one class is present in the batch/subset
        auc = 0.5

    return avg_loss, auc


def run_training(
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    patience=5,
    load_cached_data=True,
    save_path="./working/best_model.pth",
):
    """
    Orchestrates the training process with early stopping.

    Args:
        num_epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for Adam.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to use cached numpy arrays.
        save_path (str): Path to save the best model.

    Returns:
        float: The best validation AUC achieved.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Get Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    # pretrained=True ensures the backbone uses ImageNet weights
    model = RFMHDNetwork(pretrained=True)
    model.to(device)

    # 3. Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    # No weight decay as per the strategy description
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def generate_submission(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=16,
    load_cached_data=True,
):
    """
    Generates predictions for the test set using the best saved model.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached test data.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Test Loader
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Load Model
    model = RFMHDNetwork(
        pretrained=False
    )  # Pretrained weights don't matter, we load state_dict
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} does not exist. Using random weights.")

    model.to(device)
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs, batch_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions.extend(probs)
            # batch_ids is a list of strings because ids are strings
            ids.extend(batch_ids)

    # Create DataFrame
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
