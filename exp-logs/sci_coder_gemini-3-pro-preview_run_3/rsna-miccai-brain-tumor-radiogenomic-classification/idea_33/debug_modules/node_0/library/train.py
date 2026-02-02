import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.utils import seed_everything, get_device
from library.model import VAMSNet
from library.data import process_dataset, VAMSDataset

# Constants
CACHE_DIR = "./working/idea_33"
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"


def train_model(
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    debug_limit=None,
    load_cached_data=True,
    patience=5,
):
    """
    Trains the VAMSNet model with Early Stopping based on Validation AUC.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(CACHE_DIR, exist_ok=True)
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Device: {device}")

    # 2. Load Data
    # Load metadata dataframes
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))

    # Process datasets (handles caching internally)
    X_train, y_train, _ = process_dataset(
        train_df, "train", load_cached_data=load_cached_data, debug_limit=debug_limit
    )
    X_val, y_val, _ = process_dataset(
        val_df, "val", load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    # Create Datasets and Loaders
    train_dataset = VAMSDataset(X_train, y_train)
    val_dataset = VAMSDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    # drop_path_rate=0.2 as per VAMS specification
    model = VAMSNet(drop_path_rate=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_dataset)

        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_model_path


def generate_submission(model_path, batch_size=32, load_cached_data=True):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    seed_everything(42)
    device = get_device()

    # 1. Load Test Data
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # Process test data
    X_test, _, ids_test = process_dataset(
        test_df, "test", load_cached_data=load_cached_data
    )

    test_dataset = VAMSDataset(X_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Load Model
    # drop_path_rate=0.0 for inference
    model = VAMSNet(drop_path_rate=0.0).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Predict
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy().flatten())

    # 4. Save Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": predictions})

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training(
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    debug_limit=None,
    load_cached_data=True,
):
    """
    Orchestrates the training and submission pipeline.
    """
    best_model_path = train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        debug_limit=debug_limit,
        load_cached_data=load_cached_data,
    )

    generate_submission(
        model_path=best_model_path,
        batch_size=batch_size,
        load_cached_data=load_cached_data,
    )
