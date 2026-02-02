import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Performs one epoch of training.

    Args:
        model: The HPFE model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        scheduler: LR Scheduler instance.
        device: Device to train on.
        criterion: Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        # Move data to device
        cont_x = batch["cont"].to(device)
        cat_x = batch["cat"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (returns list of logits from 5 streams)
        logits_list = model(cont_x, cat_x)

        # Compute loss (Sum of BCE losses from all streams)
        loss = 0
        for logits in logits_list:
            loss += criterion(logits, target)

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

        # Scheduler step (OneCycleLR updates every batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The HPFE model.
        dataloader: Validation DataLoader.
        device: Device to evaluate on.

    Returns:
        float: ROC AUC score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)
            target = batch["target"].to(device)

            logits_list = model(cont_x, cat_x)

            # Ensemble Averaging: Mean of Sigmoids
            probs = torch.zeros_like(logits_list[0])
            for logits in logits_list:
                probs += torch.sigmoid(logits)
            probs /= len(logits_list)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    auc = roc_auc_score(all_targets, all_preds)
    return auc


def train_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping and Model Checkpointing.

    Args:
        model: The HPFE model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.

    Returns:
        model: The model loaded with the best weights.
    """
    device = Config.DEVICE
    model.to(device)

    # Optimizer: Adam with Weight Decay
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Criterion: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_auc = evaluate(model, val_loader, device)

        # Print metrics (Full precision for Val AUC as requested)
        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    return model


def predict_and_submit(model, test_loader):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained HPFE model.
        test_loader: Test DataLoader.
    """
    device = Config.DEVICE
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)
            ids = batch["id"]

            logits_list = model(cont_x, cat_x)

            # Ensemble Averaging: Mean of Sigmoids
            probs = torch.zeros_like(logits_list[0])
            for logits in logits_list:
                probs += torch.sigmoid(logits)
            probs /= len(logits_list)

            all_preds.append(probs.cpu().numpy())
            all_ids.append(ids.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    # Create submission dataframe
    submission = pd.DataFrame({Config.ID_COL: all_ids, Config.TARGET_COL: all_preds})

    # Ensure ID format is int
    submission[Config.ID_COL] = submission[Config.ID_COL].astype(int)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
