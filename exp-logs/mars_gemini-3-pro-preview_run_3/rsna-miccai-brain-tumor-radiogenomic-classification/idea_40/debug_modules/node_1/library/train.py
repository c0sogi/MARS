import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import SiameseNetwork


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The SiameseNetwork instance.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        avg_loss (float): Average training loss.
        auc (float): Training AUC score.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        # Unpack batch: SiameseDataset returns (xe, xo, label) for labeled data
        xe, xo, labels = batch
        xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Model expects (x_even, x_odd)
        logits = model(xe, xo).squeeze(1)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics tracking
        running_loss += loss.item() * xe.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge cases with single class in batch
        auc = 0.5

    return avg_loss, auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The SiameseNetwork instance.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        avg_loss (float): Average validation loss.
        auc (float): Validation AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            xe, xo, labels = batch
            xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)

            logits = model(xe, xo).squeeze(1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * xe.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def run_training(
    train_meta_path="./metadata/train.parquet",
    val_meta_path="./metadata/val.parquet",
    test_meta_path="./metadata/test.parquet",
    submission_path="./submission/submission.csv",
    cache_dir="./working/idea_40/",
    epochs=15,
    batch_size=16,
    lr=1e-4,
    seed=42,
    load_cached_data=True,
):
    """
    Orchestrates the training, validation, and inference pipeline.
    """
    # 1. Setup
    set_seed(seed)
    device = get_device()
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Data Loading
    # get_dataloaders handles caching internally via load_data_and_cache
    train_loader, val_loader, test_loader = get_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        test_meta_path=test_meta_path,
        batch_size=batch_size,
        num_workers=4,
        load_cached_data=load_cached_data,
        cache_dir=cache_dir,
    )

    # 3. Model Initialization
    model = SiameseNetwork(
        model_name="efficientnet_b0", pretrained=True, drop_path_rate=0.2
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(cache_dir, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # Train
        if train_loader:
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
        else:
            train_loss, train_auc = 0.0, 0.0

        # Validate
        if val_loader:
            val_loss, val_auc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_auc = 0.0, 0.0

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train AUC: {train_auc} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Starting inference on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    predictions = []

    if test_loader:
        with torch.no_grad():
            for batch in test_loader:
                # Test set usually returns (xe, xo) as y is None
                if len(batch) == 2:
                    xe, xo = batch
                else:
                    xe, xo, _ = batch

                xe, xo = xe.to(device), xo.to(device)
                logits = model(xe, xo).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs)

        # Retrieve IDs
        ids_path = os.path.join(cache_dir, "ids_test.npy")
        if os.path.exists(ids_path):
            test_ids = np.load(ids_path, allow_pickle=True)
        else:
            # Fallback if cache missing
            df_test = pd.read_parquet(test_meta_path)
            test_ids = df_test["BraTS21ID"].values

        # Save Submission
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
