import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, weighted_loss_metric
from library.dataset import get_dataloaders
from library.model import CervicalFractureModel


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # BCE Loss for binary classification
    # reduction='none' to handle custom weighting/aggregation if needed,
    # though here we average manually after splitting.
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["images"].to(device, non_blocking=True)  # (B, Seq, C, H, W)
        # positions = batch_data["positions"] # Unused
        targets = batch_data["targets"].to(device, non_blocking=True)  # (B, 8)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)  # (B, 7)

        # --- Hierarchical Compound Loss ---
        # 1. Vertebral Loss (C1-C7)
        # Targets cols 0-6 are C1-C7
        vert_targets = targets[:, :7]
        vert_loss = criterion(logits, vert_targets)

        # 2. Patient Overall Loss
        # Derived prediction: max probability across C1-C7
        # We use max(logits) which approximates max(sigmoid(logits)) monotonically
        # However, for loss calculation against binary target, we treat max(logits) as the logit for patient_overall
        patient_logits, _ = torch.max(logits, dim=1)
        patient_targets = targets[:, 7]
        patient_loss = criterion(patient_logits, patient_targets)

        # Total Loss: Sum of average vertebral loss and patient loss
        # This implicitly weights patient outcome heavily (1 vs 1/7 effective per vert)
        loss = vert_loss + patient_loss

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and the competition weighted metric.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["images"].to(device, non_blocking=True)
            # positions = batch_data["positions"] # Unused
            targets = batch_data["targets"].to(device, non_blocking=True)

            logits = model(images)  # (B, 7)

            # --- Loss Calculation (Same as train) ---
            vert_targets = targets[:, :7]
            vert_loss = criterion(logits, vert_targets)

            patient_logits, _ = torch.max(logits, dim=1)
            patient_targets = targets[:, 7]
            patient_loss = criterion(patient_logits, patient_targets)

            loss = vert_loss + patient_loss
            running_loss += loss.item()
            num_batches += 1

            # --- Metric Preparation ---
            # We need probabilities for the metric
            # 1. Vertebral probabilities
            probs_vert = torch.sigmoid(logits)

            # 2. Patient probability
            # The competition metric expects a specific probability for 'patient_overall'.
            # Consistent with our loss, we derive this as the max of vertebral probabilities.
            probs_patient, _ = torch.max(probs_vert, dim=1)

            # Stack to match target shape (B, 8) -> [C1...C7, patient_overall]
            # probs_patient needs to be (B, 1)
            probs_full = torch.cat([probs_vert, probs_patient.unsqueeze(1)], dim=1)

            all_preds.append(probs_full.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metric_score = weighted_loss_metric(all_preds, all_targets)
    else:
        metric_score = 0.0

    return avg_loss, metric_score


def run_training(debug=False):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists for checkpoints
    os.makedirs("working", exist_ok=True)

    # 2. Data
    train_loader, val_loader = get_dataloaders(debug=debug)
    print(
        f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # 3. Model
    model = CervicalFractureModel(pretrained=True)
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Decoupled Cosine Annealing
    # T_max is set to 1.5x epochs to prevent premature cooling
    t_max = int(Config.EPOCHS * Config.T_MAX_MULTIPLIER)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.16f} | "
            f"Val Loss: {val_loss:.16f} | "
            f"Val Metric: {val_metric:.16f}"
        )

        # Checkpoint & Early Stopping
        # We save based on Val Loss as it is the direct optimization objective proxy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "working/best_model.pth")
            print(f"New best model saved! (Loss: {val_loss:.16f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
