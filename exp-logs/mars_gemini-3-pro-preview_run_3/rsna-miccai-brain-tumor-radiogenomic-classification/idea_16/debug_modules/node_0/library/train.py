import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library import config
from library import utils
from library.model import SHDVNet
from library.data_loader import get_dataset, BraTSDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets_tensor = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets_tensor)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.extend(targets.numpy())
            all_preds.extend(probs.cpu().numpy().flatten())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    auc_score = utils.compute_auc(all_targets, all_preds)

    return epoch_loss, auc_score


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Load test data
    # Note: get_dataset handles caching logic internally
    X_test, _ = get_dataset(config.TEST_META_PATH, "test", load_cached_data=True)

    test_dataset = BraTSDataset(X_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy().flatten())

    # Load test metadata to map predictions to BraTS21ID
    if os.path.exists(config.TEST_META_PATH):
        df_test = pd.read_parquet(config.TEST_META_PATH)

        # Ensure lengths match
        if len(df_test) != len(predictions):
            print(
                f"Warning: Number of predictions ({len(predictions)}) does not match metadata rows ({len(df_test)})."
            )

        submission_df = pd.DataFrame(
            {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(f"Error: Test metadata not found at {config.TEST_META_PATH}")


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LEARNING_RATE,
    patience=5,
):
    """
    Main execution function to train the model and generate submission.
    """
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Data
    # get_dataset handles the caching logic (load if exists, else process and save)
    print("Loading training data...")
    X_train, y_train = get_dataset(
        config.TRAIN_META_PATH, "train", load_cached_data=True
    )

    print("Loading validation data...")
    X_val, y_val = get_dataset(config.VAL_META_PATH, "val", load_cached_data=True)

    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = SHDVNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    best_auc = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> New best model saved! AUC: {best_auc}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 4. Generate Submission
    if os.path.exists(config.MODEL_PATH):
        print("Loading best model for submission generation...")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        generate_submission(model, device)
    else:
        print("Error: No model checkpoint found to generate submission.")
