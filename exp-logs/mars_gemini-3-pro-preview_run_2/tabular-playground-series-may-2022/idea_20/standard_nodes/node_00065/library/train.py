import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.model import AutoencodingHybridNet


def calculate_multitask_loss(cls_logits, recon_logits, targets, recon_targets):
    """
    Computes the weighted multi-task loss.

    Args:
        cls_logits: (Batch, 1)
        recon_logits: (Batch, Seq_Len, Vocab_Size)
        targets: (Batch,)
        recon_targets: (Batch, Seq_Len)

    Returns:
        total_loss, cls_loss, recon_loss
    """
    # 1. Classification Loss (Binary Cross Entropy)
    # Targets need to be (Batch, 1) to match logits
    loss_cls_fn = nn.BCEWithLogitsLoss()
    loss_cls = loss_cls_fn(cls_logits, targets.unsqueeze(1))

    # 2. Reconstruction Loss (Cross Entropy)
    # Flatten batch and sequence dimensions for CrossEntropyLoss
    # recon_logits: (Batch * Seq_Len, Vocab_Size)
    # recon_targets: (Batch * Seq_Len)
    loss_recon_fn = nn.CrossEntropyLoss()

    batch_size, seq_len, vocab_size = recon_logits.shape
    recon_logits_flat = recon_logits.reshape(batch_size * seq_len, vocab_size)
    recon_targets_flat = recon_targets.reshape(batch_size * seq_len)

    loss_recon = loss_recon_fn(recon_logits_flat, recon_targets_flat)

    # 3. Total Loss
    total_loss = loss_cls + Config.AUX_LOSS_LAMBDA * loss_recon

    return total_loss, loss_cls, loss_recon


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        continuous = batch["continuous"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device)
        recon_targets = batch["reconstruction_target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        cls_logits, recon_logits = model(continuous, sequence)

        # Calculate loss
        loss, _, _ = calculate_multitask_loss(
            cls_logits, recon_logits, targets, recon_targets
        )

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Runs validation and computes AUC.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)
            recon_targets = batch["reconstruction_target"].to(device)

            # Forward pass
            cls_logits, recon_logits = model(continuous, sequence)

            # Calculate loss for monitoring
            loss, _, _ = calculate_multitask_loss(
                cls_logits, recon_logits, targets, recon_targets
            )
            running_loss += loss.item()

            # Apply sigmoid to classification logits for probability
            preds = torch.sigmoid(cls_logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate predictions
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Compute AUC
    auc_score = compute_auc(all_targets, all_preds)

    return avg_loss, auc_score


def predict_and_submit(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating predictions on test set...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)

            # Forward pass
            cls_logits, _ = model(continuous, sequence)

            # Apply sigmoid
            preds = torch.sigmoid(cls_logits)
            all_preds.append(preds.cpu().numpy())

    # Flatten predictions
    all_preds = np.concatenate(all_preds).flatten()

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_training():
    """
    Main training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model
    print("Initializing model...")
    model = AutoencodingHybridNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training Loop
    best_auc = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc} | LR: {current_lr:.2e} | [Saved Best]"
            )
        else:
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc} | LR: {current_lr:.2e}"
            )

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 6. Inference
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    predict_and_submit(model, test_loader, test_ids, device)
