import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import (
    BATCH_SIZE,
    EPOCHS,
    WEIGHT_DECAY,
    MAX_LR,
    PCT_START,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    ID_COL,
    TARGET_COL,
)
from library.data_utils import process_data, ManufacturingDataset, set_seed
from library.model import HPFEModel


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    Computes the sum of losses across all 5 streams.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: returns a list of 5 tensors (logits)
        stream_outputs = model(continuous, categorical)

        # Calculate total loss as sum of individual stream losses
        loss = 0
        for output in stream_outputs:
            loss += criterion(output, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * continuous.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Predictions are the average of probabilities from all 5 streams.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            stream_outputs = model(continuous, categorical)

            # Sum loss for monitoring
            loss = 0
            for output in stream_outputs:
                loss += criterion(output, targets)

            running_loss += loss.item() * continuous.size(0)

            # Aggregate predictions: Apply Sigmoid -> Mean across streams
            # stream_outputs is list of (Batch, 1) tensors
            probs_list = [torch.sigmoid(out) for out in stream_outputs]
            # Stack to (5, Batch, 1) then mean over dim 0 -> (Batch, 1)
            avg_probs = torch.stack(probs_list).mean(dim=0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(avg_probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_loss = running_loss / dataset_size

    # Calculate ROC AUC
    # Handle potential edge case with single class in batch (unlikely in full val set)
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            batch_ids = batch["id"]

            stream_outputs = model(continuous, categorical)

            # Aggregate predictions
            probs_list = [torch.sigmoid(out) for out in stream_outputs]
            avg_probs = torch.stack(probs_list).mean(dim=0)

            # Collect IDs and predictions
            # batch_ids is a Tensor due to collation
            ids.extend(batch_ids.tolist())
            preds.extend(avg_probs.cpu().numpy().flatten())

    return pd.DataFrame({ID_COL: ids, TARGET_COL: preds})


def run_training(
    epochs=EPOCHS, batch_size=BATCH_SIZE, load_cached_data=True, save_model=True
):
    """
    Main execution function to setup data, model, and run training loop.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Process Data
    train_df, val_df, test_df, vocab_sizes, cont_cols, cat_cols = process_data(
        load_cached_data=load_cached_data
    )

    # 2. Create Datasets and DataLoaders
    train_dataset = ManufacturingDataset(train_df, cont_cols, cat_cols)
    val_dataset = ManufacturingDataset(val_df, cont_cols, cat_cols)
    test_dataset = ManufacturingDataset(test_df, cont_cols, cat_cols, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = HPFEModel(vocab_sizes=vocab_sizes, num_continuous=len(cont_cols))
    model.to(device)

    # 4. Optimizer and Scheduler
    # AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MAX_LR / 10,  # Initial LR (overridden by OneCycle)
        weight_decay=WEIGHT_DECAY,
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=PCT_START,
    )

    # Loss Function (BCEWithLogitsLoss includes Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            if save_model:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Best Validation AUC: {best_auc}")

    # 6. Generate Submission
    if save_model and os.path.exists(MODEL_SAVE_PATH):
        print("Loading best model for prediction...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    print("Generating predictions on test set...")
    submission_df = predict(model, test_loader, device)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
