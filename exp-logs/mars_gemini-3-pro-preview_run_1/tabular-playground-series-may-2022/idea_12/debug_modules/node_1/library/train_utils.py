import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.model import DSDN

# -------------------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------------------


def calculate_loss(
    outputs, targets, criterion_bce, criterion_mse, criterion_ce, device
):
    """
    Computes the composite loss: BCE + Lambda * (MSE_masked + CE_masked).
    """
    logits = outputs["logits"]

    # 1. Main Task Loss (BCE with Label Smoothing)
    # Label smoothing is handled by modifying targets passed to this function or
    # if the criterion supports it. Here we do manual smoothing for BCE.
    # Smoothing: y_ls = y * (1 - alpha) + 0.5 * alpha
    smoothing = Config.LABEL_SMOOTHING
    targets_smoothed = targets * (1.0 - smoothing) + 0.5 * smoothing
    loss_bce = criterion_bce(logits.view(-1), targets_smoothed)

    # 2. Auxiliary Reconstruction Loss (Only if mask_indices exists)
    loss_recon = torch.tensor(0.0, device=device)

    mask_indices = outputs.get("mask_indices")
    if mask_indices is not None:
        # Unpack predictions and originals
        num_pred = outputs["num_pred"]  # (B, N)
        seq_pred = outputs["seq_pred"]  # (B, L, Vocab)
        num_orig = outputs["num_orig"]  # (B, N)
        seq_orig = outputs["seq_orig"]  # (B, L)

        # Determine split point for mask
        num_features = num_orig.shape[1]

        # Split mask: mask_indices is (B, N+L)
        mask_num = mask_indices[:, :num_features]
        mask_seq = mask_indices[:, num_features:]

        # Numerical Reconstruction (MSE) on masked tokens
        if mask_num.sum() > 0:
            loss_mse = criterion_mse(num_pred[mask_num], num_orig[mask_num])
        else:
            loss_mse = torch.tensor(0.0, device=device)

        # Sequence Reconstruction (CE) on masked tokens
        if mask_seq.sum() > 0:
            # Flatten predictions to (N_masked, Vocab) and targets to (N_masked)
            loss_ce = criterion_ce(seq_pred[mask_seq], seq_orig[mask_seq])
        else:
            loss_ce = torch.tensor(0.0, device=device)

        loss_recon = loss_mse + loss_ce

    # Composite Loss
    total_loss = loss_bce + Config.RECON_WEIGHT * loss_recon

    return total_loss, loss_bce.item(), loss_recon.item()


# -------------------------------------------------------------------------
# Training & Evaluation Loops
# -------------------------------------------------------------------------


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    criterion_bce,
    criterion_mse,
    criterion_ce,
):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_recon = 0.0

    for batch in dataloader:
        # Move inputs to device
        x_num = batch["numerical_features"].to(device)
        x_seq = batch["sequence_features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_num, x_seq)

        # Calculate loss
        loss, bce, recon = calculate_loss(
            outputs, targets, criterion_bce, criterion_mse, criterion_ce, device
        )

        # Backward pass
        loss.backward()

        # Optimization
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x_num.size(0)
        running_bce += bce * x_num.size(0)
        running_recon += recon * x_num.size(0)

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_bce = running_bce / dataset_size
    epoch_recon = running_recon / dataset_size

    return epoch_loss, epoch_bce, epoch_recon


def evaluate(model, dataloader, device, criterion_bce):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_num = batch["numerical_features"].to(device)
            x_seq = batch["sequence_features"].to(device)
            targets = batch["target"].to(device)

            outputs = model(x_num, x_seq)
            logits = outputs["logits"].view(-1)

            # Loss (BCE only for validation metric usually, but we can compute standard BCE)
            loss = criterion_bce(logits, targets)

            running_loss += loss.item() * x_num.size(0)

            probs = torch.sigmoid(logits)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    dataset_size = len(dataloader.dataset)
    val_loss = running_loss / dataset_size

    # Calculate AUC
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.0

    return val_loss, val_auc


def predict(model, dataloader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_num = batch["numerical_features"].to(device)
            x_seq = batch["sequence_features"].to(device)

            outputs = model(x_num, x_seq)
            logits = outputs["logits"].view(-1)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds)


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------


def run_training(train_loader, val_loader, vocab_size, num_features, seq_len):
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = DSDN(num_features=num_features, vocab_size=vocab_size, seq_len=seq_len)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # Loss Criteria
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_bce, train_recon = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion_bce,
            criterion_mse,
            criterion_ce,
        )

        val_loss, val_auc = evaluate(model, val_loader, device, criterion_bce)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} (BCE: {train_bce:.6f}, Recon: {train_recon:.6f}) | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! AUC: {best_auc:.10f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
    return model


def generate_submission(test_loader, vocab_size, num_features, seq_len, ids_test):
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = DSDN(num_features=num_features, vocab_size=vocab_size, seq_len=seq_len)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({Config.ID_COL: ids_test, Config.TARGET_COL: predictions})

    # Ensure ID is integer if needed (though sample shows it's fine)
    # The sample submission has ids like 900000, 900001

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
