import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    NUM_FOLDS,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    WORKING_DIR,
)
from library.utils import set_seed, setup_logger, AverageMeter
from library.model import RTICNN
from library.data_loader import process_data, get_fold_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch based on Dataset structure: (img, angle, label)
        inputs, angles, targets = batch

        inputs = inputs.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, angles)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, predictions (probabilities), and true labels.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in loader:
            # Handle both validation (with targets) and test (without targets) if reused
            if len(batch) == 3:
                inputs, angles, targets = batch
                targets = targets.to(device).unsqueeze(1)
                true_labels.extend(targets.cpu().numpy())
            else:
                inputs, angles = batch
                targets = None

            inputs = inputs.to(device)
            angles = angles.to(device)

            outputs = model(inputs, angles)

            if targets is not None:
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))

            # Apply Sigmoid for probabilities
            batch_preds = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(batch_preds)

    return losses.avg, np.array(preds), np.array(true_labels)


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs, angles in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            outputs = model(inputs, angles)
            batch_preds = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(batch_preds)

    return np.array(preds)


def run_training_fold(
    fold_idx, X_train, y_train, angle_train, X_test, angle_test, device, logger
):
    """
    Runs training for a single fold.
    """
    logger.info(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} ---")

    # Get DataLoaders for this fold
    train_loader, val_loader, test_loader = get_fold_loaders(
        fold_idx, X_train, y_train, angle_train, X_test, angle_test
    )

    # Initialize Model
    model = RTICNN().to(device)

    # Optimizer: AdamW with constant learning rate (no scheduler)
    # Weight decay set to 1e-2 as per Idea description
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # Loss: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)

        # Log metrics
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}"
            )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info(f"Fold {fold_idx + 1} Best Val Loss: {best_val_loss:.8f}")

    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path))

    # Generate OOF predictions
    _, val_preds, val_targets = validate(model, val_loader, criterion, device)

    # Generate Test predictions (No TTA)
    test_preds = predict_test(model, test_loader, device)

    return val_preds, val_targets, test_preds, best_val_loss


def run_training_process():
    """
    Main function to execute the full cross-validation training pipeline.
    """
    set_seed(SEED)
    logger = setup_logger(os.path.join(WORKING_DIR, "training.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    # process_data handles caching internally
    X_full, y_full, angle_full, X_test, ids_test, angle_test = process_data(
        load_cached_data=True
    )

    # Arrays to store results
    oof_preds_full = np.zeros((len(X_full), 1))
    oof_targets_full = np.zeros((len(X_full), 1))
    test_preds_accum = np.zeros((len(X_test), 1))

    # We need to reconstruct the fold indices to map OOF preds back to original indices
    # However, since get_fold_loaders does splitting internally, we need to be careful.
    # A simpler approach is to rely on the fact that StratifiedKFold is deterministic with fixed seed.
    # We will collect OOF preds and fill them into the full array based on the indices from SKF.
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        val_preds, val_targets, fold_test_preds, best_loss = run_training_fold(
            fold_idx, X_full, y_full, angle_full, X_test, angle_test, device, logger
        )

        # Store OOF predictions
        oof_preds_full[val_idx] = val_preds
        oof_targets_full[val_idx] = val_targets

        # Accumulate Test predictions
        test_preds_accum += fold_test_preds

        fold_scores.append(best_loss)

    # 2. Calculate Overall Metrics
    overall_log_loss = log_loss(y_full, oof_preds_full)
    avg_fold_loss = np.mean(fold_scores)

    logger.info("\n--- Training Complete ---")
    logger.info(f"Average Fold Best Loss: {avg_fold_loss:.8f}")
    logger.info(f"Overall OOF Log Loss: {overall_log_loss:.8f}")

    # 3. Generate Submission
    # Average predictions across folds
    avg_test_preds = test_preds_accum / NUM_FOLDS

    submission_df = pd.DataFrame(
        {"id": ids_test, "is_iceberg": avg_test_preds.flatten()}
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
