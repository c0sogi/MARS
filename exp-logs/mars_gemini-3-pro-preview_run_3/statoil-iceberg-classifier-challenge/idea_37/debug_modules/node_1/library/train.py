import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_loader import get_cv_loaders, get_test_loader
from library.model import DPDB_NBA_CNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, angles, labels in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape is (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, angles, labels in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

    return running_loss / count


def run_kfold_training(max_samples=None, num_epochs=None):
    """
    Runs the 5-Fold Cross-Validation training loop.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
        num_epochs (int, optional): Override config epochs for debugging.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS

    # Store validation scores
    fold_scores = []

    for fold in range(Config.N_FOLDS):
        print(f"Starting Fold {fold + 1}/{Config.N_FOLDS}")

        # Get DataLoaders
        train_loader, val_loader = get_cv_loaders(
            fold, load_cached_data=True, max_samples=max_samples
        )

        # Initialize Model
        model = DPDB_NBA_CNN().to(device)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = Config.get_checkpoint_path(fold)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            print(
                f"Fold {fold} Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch} for fold {fold}")
                break

        print(f"Fold {fold} Best Val Loss: {best_val_loss}")
        fold_scores.append(best_val_loss)

        # Clean up to save memory
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    avg_score = np.mean(fold_scores)
    print(f"Average CV Score (Log Loss): {avg_score}")
    return fold_scores


def generate_submission():
    """
    Loads trained models from all folds, performs inference on the test set,
    averages predictions, and saves the submission file.
    """
    print("Generating submission...")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get Test Loader
    test_loader = get_test_loader(load_cached_data=True)

    # Load Models
    models = []
    for fold in range(Config.N_FOLDS):
        model = DPDB_NBA_CNN().to(device)
        path = Config.get_checkpoint_path(fold)
        try:
            load_checkpoint(path, model, device=device)
            model.eval()
            models.append(model)
        except FileNotFoundError:
            print(f"Warning: Checkpoint for fold {fold} not found at {path}. Skipping.")

    if not models:
        raise RuntimeError("No trained models found to generate submission.")

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, angles, ids in test_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            # Ensemble Prediction
            batch_preds = []
            for model in models:
                logits = model(inputs, angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average across folds (N_folds, Batch, 1) -> (Batch, 1)
            avg_preds = np.mean(batch_preds, axis=0)

            all_preds.extend(avg_preds.flatten())
            all_ids.extend(ids)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "is_iceberg": all_preds})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
