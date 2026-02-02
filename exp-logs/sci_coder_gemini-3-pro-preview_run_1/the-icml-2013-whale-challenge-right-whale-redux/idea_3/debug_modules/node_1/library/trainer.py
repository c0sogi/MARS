import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import CRNN
from library.utils import calculate_auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: (B, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions and targets for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training(epochs=Config.EPOCHS, load_cached_data=True, debug=Config.DEBUG):
    """
    Main training loop with early stopping and checkpointing.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # Initialize Model
    print("Initializing Model...")
    model = CRNN().to(device)

    # Loss function with class imbalance handling
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    best_val_auc = 0.0
    patience_counter = 0

    # Ensure working directory exists for checkpoint
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    print("Starting Training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Step scheduler based on Validation AUC
        scheduler.step(val_auc)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed}s | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (AUC: {best_val_auc})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model weights before returning
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    return model, test_loader


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating Submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())

    # Load test metadata to get clip IDs
    df_test = pd.read_csv(Config.TEST_CSV)

    # Safety check for length mismatch
    if len(df_test) != len(all_preds):
        print(
            f"Warning: Mismatch in predictions length. Metadata: {len(df_test)}, Preds: {len(all_preds)}"
        )
        min_len = min(len(df_test), len(all_preds))
        df_test = df_test.iloc[:min_len]
        all_preds = all_preds[:min_len]

    df_test["probability"] = all_preds

    # Format submission: clip, probability
    submission_df = df_test[["clip", "probability"]]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
