import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import DeGUTModel


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using the composite denoising loss.
    """
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_recon = 0.0

    # Loss functions
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        num_features = batch["num_features"].to(device)
        seq_features = batch["seq_features"].to(device)
        mask_num = batch["mask_num"].to(device)
        mask_seq = batch["mask_seq"].to(device)

        target_cls = batch["target_cls"].to(device)
        target_num = batch["target_num"].to(device)
        target_seq = batch["target_seq"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            num_features=num_features,
            seq_features=seq_features,
            mask_num=mask_num,
            mask_seq=mask_seq,
        )

        logits_cls = outputs["logits_cls"]
        pred_num = outputs["pred_num"]
        pred_seq = outputs["pred_seq"]

        # --- Calculate Losses ---

        # 1. Supervised Classification Loss
        loss_bce = criterion_bce(logits_cls, target_cls.unsqueeze(1))

        # 2. Reconstruction Loss (Numerical)
        # Only compute loss on masked values
        if mask_num.sum() > 0:
            loss_mse = criterion_mse(pred_num[mask_num], target_num[mask_num])
        else:
            loss_mse = torch.tensor(0.0, device=device)

        # 3. Reconstruction Loss (Sequence)
        # Only compute loss on masked tokens
        if mask_seq.sum() > 0:
            # Flatten predictions to (N_masked, Vocab) and targets to (N_masked)
            masked_pred_seq = pred_seq[mask_seq]
            masked_target_seq = target_seq[mask_seq]
            loss_ce = criterion_ce(masked_pred_seq, masked_target_seq)
        else:
            loss_ce = torch.tensor(0.0, device=device)

        # Composite Loss
        loss_recon = loss_mse + loss_ce
        loss = loss_bce + Config.LOSS_LAMBDA * loss_recon

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for Transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_recon += loss_recon.item()

    avg_loss = total_loss / len(dataloader)
    avg_bce = total_bce / len(dataloader)
    avg_recon = total_recon / len(dataloader)

    print(
        f"Epoch {epoch+1} | Train Loss: {avg_loss:.5f} (BCE: {avg_bce:.5f}, Recon: {avg_recon:.5f})"
    )

    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on validation data (no masking).
    Returns the ROC AUC score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            num_features = batch["num_features"].to(device)
            seq_features = batch["seq_features"].to(device)
            target_cls = batch["target_cls"].to(device)

            # Forward pass (collator ensures masks are all False for val/test)
            outputs = model(
                num_features=num_features,
                seq_features=seq_features,
                mask_num=None,
                mask_seq=None,
            )

            logits = outputs["logits_cls"]
            probs = torch.sigmoid(logits).squeeze()

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(target_cls.cpu().numpy())

    auc = compute_metric(all_targets, all_preds)
    return auc


def run_training():
    """
    Main execution function for the training pipeline.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Data
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
    )

    # Determine input dimensions from a batch
    sample_batch = next(iter(train_loader))
    num_feats = sample_batch["num_features"].shape[1]
    vocab_size = len(vocab)

    print(f"Data Loaded. Num Features: {num_feats}, Vocab Size: {vocab_size}")

    # 2. Initialize Model
    device = torch.device(Config.DEVICE)
    model = DeGUTModel(num_feats=num_feats, vocab_size=vocab_size)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.10f}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved! AUC: {val_auc:.10f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    # 5. Generate Submission
    predict_and_submit(model, test_loader, device)


def predict_and_submit(model, test_loader, device):
    """
    Loads the best model, generates predictions on the test set,
    and saves the submission file.
    """
    print("Generating predictions for test set...")

    # Load best weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No saved model found. Using current weights.")

    model.eval()

    all_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            num_features = batch["num_features"].to(device)
            seq_features = batch["seq_features"].to(device)
            ids = batch["ids"]

            outputs = model(
                num_features=num_features,
                seq_features=seq_features,
                mask_num=None,
                mask_seq=None,
            )

            logits = outputs["logits_cls"]
            probs = torch.sigmoid(logits).squeeze()

            # Handle single-item batch edge case where squeeze returns 0-d tensor
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "target": all_probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(df_sub.head())
