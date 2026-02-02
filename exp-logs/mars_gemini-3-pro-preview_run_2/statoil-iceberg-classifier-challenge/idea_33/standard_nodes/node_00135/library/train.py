import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import MDSWBN
from library.data_loader import get_data, get_loaders, get_test_loader
from library.utils import seed_everything


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    return running_loss / dataset_size


def run_fold(fold_idx, X, y, inc_angles, epochs=Config.EPOCHS, debug=False):
    """
    Runs training for a specific fold.
    """
    print(f"Starting Fold {fold_idx + 1}")

    # Get loaders for this fold
    train_loader, val_loader = get_loaders(fold_idx, X, y, inc_angles)

    # Initialize Model
    model = MDSWBN().to(Config.DEVICE)

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.BCELoss()

    # Training Loop Variables
    best_loss = float("inf")
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model for this fold
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
            torch.save(best_model_wts, save_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered")
            break

    return best_model_wts


def predict_fold(model, test_loader, device):
    """
    Generates predictions for the test set using a trained model.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            angles = batch[1].to(device)
            outputs = model(images, angles)
            preds.extend(outputs.cpu().numpy())
    return np.array(preds)


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Orchestrates the full 5-fold cross-validation training and submission generation.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Data
    # get_data handles caching and processing
    X, y, inc, X_test, inc_test, test_ids = get_data(debug=debug)

    test_preds_accumulator = np.zeros((len(X_test), 1))

    # Iterate through folds
    for fold in range(Config.N_FOLDS):
        # Train the fold
        best_wts = run_fold(fold, X, y, inc, epochs=epochs, debug=debug)

        # Load best weights for inference
        model = MDSWBN().to(Config.DEVICE)
        model.load_state_dict(best_wts)

        # Predict on Test Set
        test_loader = get_test_loader(X_test, inc_test)
        fold_preds = predict_fold(model, test_loader, Config.DEVICE)

        test_preds_accumulator += fold_preds

    # Average predictions across folds
    avg_preds = test_preds_accumulator / Config.N_FOLDS

    # Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds.flatten()})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
