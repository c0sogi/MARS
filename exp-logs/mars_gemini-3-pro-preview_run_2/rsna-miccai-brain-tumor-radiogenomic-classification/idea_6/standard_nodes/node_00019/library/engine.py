import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    SUBMISSION_PATH,
    BATCH_SIZE,
)
from library.utils import save_checkpoint, load_checkpoint
from library.model import GroupedEfficientNet
from library.data_loader import get_dataloader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for AUC calculation
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle case with single class in batch or other edge cases
    # Cite debug_lesson_1: Guard against single-class subsets returning nan
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
            if np.isnan(epoch_auc):
                epoch_auc = 0.5
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Cite debug_lesson_1: Guard against single-class subsets returning nan
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
            if np.isnan(epoch_auc):
                epoch_auc = 0.5
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(train_loader, val_loader):
    """
    Main training loop with Early Stopping.
    """
    model = GroupedEfficientNet(pretrained=True).to(DEVICE)

    # AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        # Print full precision metrics
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss} | Train AUC: {train_auc} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs of no improvement."
            )
            break

    return model


def predict(model, loader, device):
    """
    Runs inference with Test-Time Augmentation (TTA).
    Returns a list of probabilities corresponding to the loader's order.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # TTA 1: Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # TTA 2: Horizontal Flip (dim 3 for N, C, H, W)
            images_h = torch.flip(images, [3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # TTA 3: Vertical Flip (dim 2 for N, C, H, W)
            images_v = torch.flip(images, [2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average predictions
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            all_probs.extend(avg_prob.cpu().numpy().flatten().tolist())

    return all_probs


def generate_submission(test_df, model_path=MODEL_SAVE_PATH):
    """
    Generates the submission file using the best saved model.
    """
    # Load Model
    model = GroupedEfficientNet(pretrained=False).to(DEVICE)
    load_checkpoint(model_path, model, device=DEVICE)

    # Create Test Loader (shuffle=False to preserve order)
    test_loader = get_dataloader(
        test_df, batch_size=BATCH_SIZE, phase="test", load_cached_data=True
    )

    # Run Inference
    probs = predict(model, test_loader, DEVICE)

    # Create Submission DataFrame
    # Note: The loader preserves the order of test_df
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": probs}
    )

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
