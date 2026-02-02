import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import get_dataloaders
from library.model import WideSEResNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
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
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def train_fold(fold_idx):
    """
    Trains a model for a specific fold with Early Stopping.
    """
    print(f"\n--- Starting Fold {fold_idx} ---")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(fold_idx=fold_idx)

    # Initialize Model
    model = WideSEResNet().to(device)

    # Loss and Optimizer
    # Using BCEWithLogitsLoss as model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer with constant learning rate (no scheduler)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Config.get_checkpoint_path(fold_idx)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss


def inference(fold_idx, test_loader):
    """
    Generates predictions for the test set using the trained model from a specific fold.
    Disables TTA (Test-Time Augmentation).
    """
    # Initialize model structure
    model = WideSEResNet().to(device)

    # Load weights
    checkpoint_path = Config.get_checkpoint_path(fold_idx)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for fold {fold_idx}: {checkpoint_path}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = model(images, angles)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    return np.concatenate(all_probs), all_ids


def run_training_pipeline():
    """
    Orchestrates the full 5-Fold Cross-Validation training and submission generation.
    """
    set_seed(Config.SEED)
    Config.setup()

    # Store predictions from each fold
    fold_predictions = []
    test_ids = None

    # Loop through all folds
    for fold in range(Config.NUM_FOLDS):
        # Train
        train_fold(fold)

        # Inference
        # We need the test loader. Since it's the same for all folds, getting it from any fold call works.
        _, _, test_loader = get_dataloaders(fold_idx=fold, load_cached_data=True)

        print(f"Generating predictions for Fold {fold}...")
        probs, ids = inference(fold, test_loader)

        fold_predictions.append(probs)

        if test_ids is None:
            test_ids = ids
        else:
            # Sanity check to ensure ID order is preserved
            if list(test_ids) != list(ids):
                raise ValueError(f"Test ID mismatch in fold {fold}")

    # Ensemble: Simple Average
    print("\nAggregating predictions...")
    avg_preds = np.mean(fold_predictions, axis=0)

    # Save Submission
    save_submission(test_ids, avg_preds, Config.SUBMISSION_FILE)
    print("Pipeline completed successfully.")
