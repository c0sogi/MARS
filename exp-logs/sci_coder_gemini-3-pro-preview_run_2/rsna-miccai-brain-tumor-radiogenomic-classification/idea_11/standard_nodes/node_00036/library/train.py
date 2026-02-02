import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data import get_dataloader, set_seed
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and labels for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC, handling potential edge cases (e.g., single class in batch)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Main training loop with early stopping.
    """
    print(f"Initializing AsymmetricEfficientNet on {Config.DEVICE}...")
    model = AsymmetricEfficientNet().to(Config.DEVICE)

    # Signal-Preserving Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = get_dataloader("train", debug=Config.DEBUG)
    val_loader = get_dataloader("val", debug=Config.DEBUG)

    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss}, Train AUC: {train_auc} - "
            f"Val Loss: {val_loss}, Val AUC: {val_auc} - "
            f"Time: {elapsed}s"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"  [New Best Model] Saved to {Config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc}")
    return model


def predict_and_submit():
    """
    Runs inference on the test set using Test-Time Augmentation (TTA) and generates submission file.
    """
    print("Starting Inference with TTA...")

    # Load Best Model
    model = AsymmetricEfficientNet().to(Config.DEVICE)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded weights from {Config.CHECKPOINT_PATH}")
    else:
        print("Warning: Checkpoint not found. Using random weights (for debugging).")

    model.eval()

    test_loader = get_dataloader(
        "test", debug=Config.DEBUG, batch_size=Config.BATCH_SIZE
    )

    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            # images shape: (B, 12, H, W)
            images = images.to(Config.DEVICE)

            # TTA: Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA: Horizontal Flip
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # TTA: Vertical Flip
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average Predictions
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            predictions.extend(avg_probs.cpu().numpy().flatten())

    # Load test metadata to get IDs
    df_test = pd.read_csv(Config.TEST_METADATA)
    if Config.DEBUG:
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Ensure lengths match
    if len(predictions) != len(df_test):
        print(
            f"Warning: Prediction count {len(predictions)} != Test ID count {len(df_test)}"
        )
        min_len = min(len(predictions), len(df_test))
        predictions = predictions[:min_len]
        df_test = df_test.iloc[:min_len]

    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    set_seed(Config.SEED)
    run_training()
    predict_and_submit()
