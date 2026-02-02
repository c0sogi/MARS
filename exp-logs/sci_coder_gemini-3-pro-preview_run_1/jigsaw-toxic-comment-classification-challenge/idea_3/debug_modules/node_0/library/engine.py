import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import get_dataloaders
from library.model import ToxicityModel


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_fn(model, dataloader, optimizer, scheduler, device, loss_fn, scaler):
    """
    Training loop for one epoch.
    """
    model.train()
    final_loss = 0
    count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=True):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        final_loss += loss.item()
        count += 1

    return final_loss / count


def eval_fn(model, dataloader, device, loss_fn):
    """
    Evaluation loop for validation set.
    """
    model.eval()
    final_loss = 0
    count = 0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mixed precision is generally not needed for inference accuracy but can speed it up
            # We stick to standard float32 for stable eval metric calculation if memory allows
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            final_loss += loss.item()
            count += 1

            # Store logits and labels for AUC calculation
            preds.append(logits.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = final_loss / count

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Apply sigmoid to logits for probability-based metrics if needed,
    # but roc_auc_score works with probabilities or ranked scores (logits are monotonic w.r.t probs)
    # We apply sigmoid to be safe and consistent with interpretation
    preds_probs = 1 / (1 + np.exp(-preds))

    # Calculate Mean Column-wise ROC AUC
    # average='macro' computes the metric for each label, and finds their unweighted mean
    try:
        roc_auc = roc_auc_score(targets, preds_probs, average="macro")

        # Calculate individual column AUCs for detailed logging if needed
        col_aucs = roc_auc_score(targets, preds_probs, average=None)
    except ValueError as e:
        print(f"Error calculating ROC AUC: {e}")
        roc_auc = 0.0
        col_aucs = []

    return avg_loss, roc_auc, col_aucs


def predict_fn(model, dataloader, device):
    """
    Prediction loop for test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds)


def run_training():
    """
    Main function to run the training pipeline.
    """
    config = Config()
    set_seed(config.seed)

    # Ensure directories exist
    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(config.submission_dir, exist_ok=True)

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print("Initializing Model...")
    model = ToxicityModel()
    model.to(config.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Scheduler
    # OneCycleLR requires total steps
    total_steps = len(train_loader) * config.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.pct_start,
    )

    # Loss Function
    loss_fn = nn.BCEWithLogitsLoss()

    # GradScaler for Mixed Precision
    scaler = GradScaler()

    best_auc = -1.0
    patience = 0
    early_stopping_patience = (
        2  # Stop if no improvement for 2 epochs (after initial improvement)
    )

    print("Starting Training...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, config.device, loss_fn, scaler
        )
        print(f"Train Loss: {train_loss}")

        # Evaluate
        val_loss, val_auc, col_aucs = eval_fn(model, val_loader, config.device, loss_fn)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Mean AUC: {val_auc}")
        print(f"Column-wise AUCs: {col_aucs}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            print(
                f"Validation AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), config.model_save_path)
            patience = 0
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{early_stopping_patience}")
            if patience >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load Best Model for Inference
    print("\nLoading best model for inference...")
    if os.path.exists(config.model_save_path):
        model.load_state_dict(
            torch.load(config.model_save_path, map_location=config.device)
        )
    else:
        print("Warning: No model file found. Using current model state.")

    # Predict on Test Set
    print("Generating predictions on test set...")
    test_probs = predict_fn(model, test_loader, config.device)

    # Prepare Submission
    print("Saving submission...")
    # Load test IDs from metadata to ensure correct order
    test_meta = pd.read_csv(config.test_metadata_path)

    submission_df = pd.DataFrame(test_probs, columns=config.labels)
    submission_df.insert(0, "id", test_meta["id"])

    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
