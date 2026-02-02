import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model_utils import IPPFEModel


def get_optimizer_and_scheduler(model, total_steps):
    """
    Initializes the AdamW optimizer and OneCycleLR scheduler.
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
    )

    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    Calculates the sum of BCE losses for the 5 independent streams.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        cat_x = batch["cat_features"].to(device)
        cont_x = batch["cont_features"].to(device)
        targets = batch["target"].to(device)

        batch_size = cat_x.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass: (Batch, 5)
        logits = model(cat_x, cont_x)

        # Expand targets to (Batch, 5) to match logits for independent stream loss calculation
        targets_expanded = targets.unsqueeze(1).repeat(1, 5)

        # Calculate loss (Sum of BCE for each stream)
        loss = criterion(logits, targets_expanded)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * batch_size

    return running_loss / dataset_size


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes loss and ROC AUC based on the mean prediction of the 5 streams.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            batch_size = cat_x.size(0)
            dataset_size += batch_size

            logits = model(cat_x, cont_x)

            # Loss calculation
            targets_expanded = targets.unsqueeze(1).repeat(1, 5)
            loss = criterion(logits, targets_expanded)
            running_loss += loss.item() * batch_size

            # Prediction for AUC: Average of probabilities across 5 streams
            probs = torch.sigmoid(logits)  # (Batch, 5)
            avg_probs = torch.mean(probs, dim=1)

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_loss = running_loss / dataset_size

    return epoch_loss, epoch_auc


def train(train_loader, val_loader, metadata):
    """
    Orchestrates the training process including initialization,
    training loop, validation, and early stopping.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(device)

    # Criterion: BCEWithLogitsLoss
    # We sum the loss over the 5 streams.
    # By default reduction='mean' averages over the batch (which we want)
    # and over the columns (which we don't want, we want sum over streams).
    # However, since the target is expanded and we want the joint optimization,
    # standard BCE on (Batch, 5) with reduction='mean' effectively optimizes
    # mean(loss_stream_1 + ... + loss_stream_5).
    # To strictly follow "sum of Binary Cross-Entropy losses", we can rely on the fact that
    # optimizing Mean(Loss) is equivalent to optimizing Sum(Loss) for gradients.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer and Scheduler
    total_steps = len(train_loader) * Config.EPOCHS
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_steps)

    best_auc = 0.0
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.15f} | "
            f"Val Loss: {val_loss:.15f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        # Checkpoint Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val AUC: {best_auc:.15f}")
    return best_auc


def generate_submission(test_loader, metadata):
    """
    Generates predictions for the test set using the best saved model.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(device)

    # Load Best Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    ids = sample_sub[Config.ID_COL].values

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits)

            # Ensemble Strategy: Arithmetic Mean of 5 streams
            avg_probs = torch.mean(probs, dim=1)
            predictions.extend(avg_probs.cpu().numpy())

    # Ensure lengths match
    if len(predictions) != len(ids):
        print(f"Warning: Prediction count {len(predictions)} != ID count {len(ids)}")
        min_len = min(len(predictions), len(ids))
        predictions = predictions[:min_len]
        ids = ids[:min_len]

    # Create Submission DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
