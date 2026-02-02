import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import EarlyStopping, seed_everything
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import DPCNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, inc_angles, labels in loader:
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Match output shape (B, 1)

        optimizer.zero_grad()

        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        float: Average validation loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, inc_angles, labels in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return val_loss


def train_fold(fold_idx):
    """
    Trains a model for a specific cross-validation fold.
    Manages optimizer, scheduler, and early stopping.
    Saves the best model weights to disk.

    Args:
        fold_idx (int): Index of the fold (0 to NUM_FOLDS-1).

    Returns:
        float: Best validation loss achieved.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"\n--- Starting Training for Fold {fold_idx} ---")

    # Get DataLoaders
    train_loader, val_loader = get_fold_loaders(fold_idx)

    # Initialize Model
    model = DPCNet().to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="min")

    best_loss = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if val_loss < best_loss:
            best_loss = val_loss

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save the best model state
    save_path = os.path.join(Config.ARTIFACT_DIR, f"model_fold_{fold_idx}.pth")
    # Restore best weights from memory before saving
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), save_path)
    print(
        f"Fold {fold_idx} finished. Best Val Loss: {early_stopping.best_score:.8f}. Model saved to {save_path}"
    )

    return early_stopping.best_score


def run_cross_validation():
    """
    Runs training for all folds sequentially.
    """
    scores = []
    for fold in range(Config.NUM_FOLDS):
        score = train_fold(fold)
        scores.append(score)

    print("\n--- Cross-Validation Results ---")
    for i, score in enumerate(scores):
        print(f"Fold {i}: {score:.8f}")
    print(f"Average Log Loss: {np.mean(scores):.8f}")


def predict_ensemble():
    """
    Loads all trained fold models, performs inference on the test set,
    averages the predictions, and generates the submission file.
    """
    print("\n--- Starting Ensemble Prediction ---")
    device = torch.device(Config.DEVICE)

    # Get Test Loader
    test_loader = get_test_loader()

    # Prepare to collect predictions
    # We need to map IDs to predictions.
    # Since DataLoader order is deterministic if shuffle=False and num_workers=0 or fixed seed,
    # we can just accumulate predictions in a list and match with IDs returned by the loader.

    # However, to be safe with ensemble, we will accumulate probabilities for each sample
    # and then average.

    # First, get all IDs and initialize prediction array
    all_ids = []
    # We'll run one pass to get IDs and size (or rely on loader size if known, but safer to iterate)
    # Actually, we can just iterate the loader for each model and sum the predictions.

    # Placeholder for summed predictions
    # We don't know exact size yet, so we'll collect lists of tensors first

    fold_predictions = []

    for fold_idx in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.ARTIFACT_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for fold {fold_idx}...")
        model = DPCNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_probs = []
        ids_list = []

        with torch.no_grad():
            for images, inc_angles, img_ids in test_loader:
                images = images.to(device)
                inc_angles = inc_angles.to(device)

                outputs = model(images, inc_angles)
                probs = outputs.cpu().numpy().flatten()

                fold_probs.extend(probs)

                # Only need to collect IDs once, but good to verify consistency
                if fold_idx == 0:
                    ids_list.extend(img_ids)

        fold_predictions.append(np.array(fold_probs))
        if fold_idx == 0:
            all_ids = ids_list

    if not fold_predictions:
        print("No models found to generate predictions.")
        return

    # Convert to numpy array: (Num_Folds, Num_Samples)
    fold_predictions = np.array(fold_predictions)

    # Average predictions
    avg_predictions = np.mean(fold_predictions, axis=0)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "is_iceberg": avg_predictions})

    # Save submission
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())
