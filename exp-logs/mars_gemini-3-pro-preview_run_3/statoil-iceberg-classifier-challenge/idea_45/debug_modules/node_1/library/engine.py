import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import AverageMeter, save_checkpoint
from library.model import BDPH_CNN


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    losses = AverageMeter()
    model.train()

    for i, (images, angles, targets) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}/{Config.EPOCHS}] Training Loss: {losses.avg}")
    return losses.avg


def validate_one_epoch(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    # Calculate Log Loss using sklearn
    if len(all_targets) > 0:
        val_log_loss = log_loss(all_targets, all_preds, labels=[0, 1])
    else:
        val_log_loss = losses.avg

    print(f"Validation Loss (BCE): {losses.avg}")
    print(f"Validation Log Loss: {val_log_loss}")

    return val_log_loss


def train_fold(fold, train_loader, val_loader):
    """
    Orchestrates the training for a single fold.
    Initializes model, optimizer, and handles the training loop with early stopping.
    """
    device = torch.device(Config.DEVICE)
    print(f"--- Starting Fold {fold} on {device} ---")

    # Initialize Model
    model = BDPH_CNN().to(device)

    # Optimizer: AdamW with constant learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_score = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)

        # Validate
        val_score = validate_one_epoch(val_loader, model, criterion, device)

        # Checkpoint & Early Stopping
        is_best = val_score < best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            print(f"New Best Score for Fold {fold}: {best_score}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Save Checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "fold": fold,
            },
            is_best,
            fold,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_score


def generate_submission(test_loader):
    """
    Generates predictions for the test set using an ensemble of models from all folds.
    Saves the result to submission.csv.
    """
    device = torch.device(Config.DEVICE)
    model = BDPH_CNN().to(device)

    ensemble_probs = None
    test_ids = []

    print("Generating submission with ensemble...")

    # Iterate over all folds
    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Loading checkpoint for fold {fold}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        fold_probs = []
        current_ids = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                fold_probs.extend(probs)
                if fold == 0:
                    current_ids.extend(ids)

        fold_probs = np.array(fold_probs)

        if ensemble_probs is None:
            ensemble_probs = fold_probs
            test_ids = current_ids
        else:
            ensemble_probs += fold_probs

    if ensemble_probs is None:
        raise RuntimeError("No checkpoints found. Cannot generate submission.")

    # Average probabilities
    avg_probs = ensemble_probs / Config.N_FOLDS

    # Create submission dataframe
    df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
