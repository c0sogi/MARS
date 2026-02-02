import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_device
from library.model import ManufacturingMLP


def get_vocab_sizes(train_loader, val_loader, test_loader):
    """
    Calculates the vocabulary size for each categorical feature by inspecting
    the maximum index present across Train, Validation, and Test datasets.
    This ensures the Embedding layers are sized correctly for the transductive encoding.
    """
    train_cat = train_loader.dataset.cat_features
    val_cat = val_loader.dataset.cat_features
    test_cat = test_loader.dataset.cat_features

    # Concatenate all to find global max indices
    all_cat = np.vstack([train_cat, val_cat, test_cat])

    # Vocab size is max_index + 1
    vocab_sizes = (all_cat.max(axis=0) + 1).tolist()
    return vocab_sizes


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        x_cat = batch["x_cat"].to(device)
        x_cont = batch["x_cont"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        batch_size = x_cat.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        logits = model(x_cat, x_cont)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_cat = batch["x_cat"].to(device)
            x_cont = batch["x_cont"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            batch_size = x_cat.size(0)
            dataset_size += batch_size

            logits = model(x_cat, x_cont)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Fallback if only one class is present in the validation batch (unlikely)
        auc_score = 0.5

    return epoch_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_cat = batch["x_cat"].to(device)
            x_cont = batch["x_cont"].to(device)

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training(train_loader, val_loader, test_loader):
    """
    Main driver function to setup model, optimizer, scheduler and run the training loop.
    """
    device = get_device()
    print(f"Training on device: {device}")

    # 1. Determine Vocabulary Sizes
    vocab_sizes = get_vocab_sizes(train_loader, val_loader, test_loader)

    # 2. Initialize Model
    model = ManufacturingMLP(vocab_sizes).to(device)

    # 3. Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    return model


def generate_submission(model, test_loader):
    """
    Generates predictions using the best model and saves the submission CSV.
    """
    device = get_device()
    print("Generating predictions for test set...")

    # Generate probabilities
    preds = predict(model, test_loader, device)

    # Load Test IDs from source to ensure alignment
    # We read directly from the metadata/test.csv file
    df_test = pd.read_csv(Config.TEST_DATA_PATH)
    test_ids = df_test[Config.ID_COL].values

    if len(test_ids) != len(preds):
        print(
            f"Warning: ID count ({len(test_ids)}) does not match Prediction count ({len(preds)})."
        )

    # Create DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: preds})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
