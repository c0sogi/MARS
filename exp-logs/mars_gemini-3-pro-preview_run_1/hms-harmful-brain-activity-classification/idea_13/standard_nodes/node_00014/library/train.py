import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.data import get_dataloaders
from library.models import AuxiliaryFusionNet
from library.utils import seed_everything, kl_divergence


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using Multi-Task KL Divergence Loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # KLDivLoss expects input to be log-probabilities and target to be probabilities
    criterion = nn.KLDivLoss(reduction="batchmean")

    # Use tqdm for progress tracking
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, (eeg, spec, targets) in enumerate(pbar):
        eeg = eeg.to(device)
        spec = spec.to(device)
        targets = targets.to(device)

        batch_size = eeg.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass: Returns joint logits and auxiliary logits
        joint_logits, aux_eeg_logits, aux_spec_logits = model(eeg, spec)

        # Apply LogSoftmax for KLDivLoss
        joint_log_probs = F.log_softmax(joint_logits, dim=1)
        aux_eeg_log_probs = F.log_softmax(aux_eeg_logits, dim=1)
        aux_spec_log_probs = F.log_softmax(aux_spec_logits, dim=1)

        # Calculate individual losses
        loss_joint = criterion(joint_log_probs, targets)
        loss_aux_eeg = criterion(aux_eeg_log_probs, targets)
        loss_aux_spec = criterion(aux_spec_log_probs, targets)

        # Weighted Sum (Multi-Task Loss)
        total_loss = loss_joint + Config.AUX_LOSS_WEIGHT * (
            loss_aux_eeg + loss_aux_spec
        )

        # Backward pass
        total_loss.backward()

        # Optimizer and Scheduler step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += total_loss.item() * batch_size
        dataset_size += batch_size

        pbar.set_postfix({"loss": total_loss.item()})

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Validates the model using the Joint Head predictions and KL Divergence metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    # Criterion for validation loss tracking (Joint head only)
    criterion = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Val]", leave=False)
        for eeg, spec, targets in pbar:
            eeg = eeg.to(device)
            spec = spec.to(device)
            targets = targets.to(device)

            batch_size = eeg.size(0)

            # Forward pass
            joint_logits, _, _ = model(eeg, spec)

            # Calculate Loss
            joint_log_probs = F.log_softmax(joint_logits, dim=1)
            loss = criterion(joint_log_probs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store probabilities for metric calculation
            probs = F.softmax(joint_logits, dim=1)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Calculate global metric using the provided utility
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    val_metric = kl_divergence(all_targets, all_preds)

    return val_loss, val_metric


def inference(model, loader, device):
    """
    Generates predictions for the test set using the best model.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Test]", leave=False)
        for eeg, spec in pbar:
            eeg = eeg.to(device)
            spec = spec.to(device)

            # Forward pass
            joint_logits, _, _ = model(eeg, spec)

            # Softmax to get probabilities
            probs = F.softmax(joint_logits, dim=1)
            preds_list.append(probs.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def train(debug_limit=None, epochs=Config.EPOCHS):
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_limit=debug_limit,
    )

    # 2. Model
    print(f"Initializing {Config.MODEL_NAME}...")
    model = AuxiliaryFusionNet().to(device)

    # 3. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 4. Training Loop
    best_metric = float("inf")
    patience = 5
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Print metrics (Full precision for KL)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val KL: {val_metric}"
        )

        # Checkpointing & Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! KL: {val_metric}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Submission
    print("\nGenerating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Predict
    test_preds = inference(model, test_loader, device)

    # Prepare Submission DataFrame
    # Load test metadata to get correct eeg_id order
    test_df = pd.read_csv(Config.TEST_CSV)

    # Handle debug case where loader might be shorter than file
    if len(test_df) != len(test_preds):
        test_df = test_df.iloc[: len(test_preds)]

    submission_df = pd.DataFrame(
        {
            "eeg_id": test_df["eeg_id"],
            "seizure_vote": test_preds[:, 0],
            "lpd_vote": test_preds[:, 1],
            "gpd_vote": test_preds[:, 2],
            "lrda_vote": test_preds[:, 3],
            "grda_vote": test_preds[:, 4],
            "other_vote": test_preds[:, 5],
        }
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
