import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.utils import set_seed
from library.data_loader import load_data, IcebergDataset, get_transforms
from library.model import SimpleCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The DataLoader for training data.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for imgs, angles, labels in loader:
        imgs = imgs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The DataLoader for validation data.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for imgs, angles, labels in loader:
            imgs = imgs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

    return running_loss / dataset_size


def run_training_pipeline(epochs=50, batch_size=32, patience=10, seed=42):
    """
    Executes the 5-Fold Cross-Validation training pipeline and generates the submission.

    Args:
        epochs (int): Maximum number of epochs per fold.
        batch_size (int): Batch size for dataloaders.
        patience (int): Early stopping patience.
        seed (int): Random seed.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data using the provided data_loader library
    # load_data handles caching and imputation
    data = load_data(load_cached_data=True)

    # Merge train and validation sets provided by loader to perform stratified K-Fold
    # We do this because the provided metadata split is 80/20, but we want to use
    # 5-Fold CV on the entire labeled dataset for better robustness.
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Accumulator for test predictions (probabilities)
    test_preds_accum = np.zeros(len(X_test))

    # Prepare Test Loader
    test_dataset = IcebergDataset(X_test, angle_test, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Starting 5-Fold Cross-Validation on {len(X_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nFold {fold + 1}/5")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        angle_train_fold, angle_val_fold = angle_full[train_idx], angle_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold,
            angle_train_fold,
            y_train_fold,
            transform=get_transforms("train"),
        )
        val_dataset = IcebergDataset(
            X_val_fold, angle_val_fold, y_val_fold, transform=get_transforms("test")
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = SimpleCNN().to(device)

        # Optimizer: Adam with constant LR
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Loss: BCEWithLogitsLoss (combines Sigmoid + BCELoss)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}"
            )

            # Check Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Inference on Test Set for this Fold
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for imgs, angles in test_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                outputs = model(imgs, angles)
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        test_preds_accum += np.array(fold_preds)

    # Ensemble and Submission
    # Average probabilities across 5 folds
    avg_preds = test_preds_accum / 5.0

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    submission.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
