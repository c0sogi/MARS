import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from library.config import (
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    BG_CLASS_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    WORKING_DIR,
    set_seed,
)
from library.utils import decode_predictions, compute_levenshtein_ratio
from library.data_loader import get_data_loaders
from library.model import CascadedRefinementNet


def compute_loss(s1_logits, s2_logits, targets, criterion_ce):
    """
    Computes the combined loss for the Cascaded Refinement Network.

    Args:
        s1_logits: Output from Stage 1 (Batch, Time, NumClasses)
        s2_logits: Output from Stage 2 (Batch, Time, NumClasses)
        targets: Ground truth labels (Batch, Time)
        criterion_ce: Weighted CrossEntropyLoss instance

    Returns:
        total_loss, loss_s1, loss_s2, loss_smooth
    """
    # Flatten for CrossEntropy: (Batch * Time, NumClasses) vs (Batch * Time)
    batch_size, time_steps, num_classes = s1_logits.shape

    s1_flat = s1_logits.reshape(-1, num_classes)
    s2_flat = s2_logits.reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)

    # 1. Classification Losses
    loss_s1 = criterion_ce(s1_flat, targets_flat)
    loss_s2 = criterion_ce(s2_flat, targets_flat)

    # 2. Smoothing Loss (MSE on log-probs of adjacent frames for Stage 2)
    # Get log probabilities
    log_probs = F.log_softmax(s2_logits, dim=2)

    # Calculate difference between t and t-1
    # shape: (Batch, Time-1, NumClasses)
    diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

    # MSE of differences (encourage them to be 0)
    loss_smooth = torch.mean(diff**2)

    # Combined Loss
    # We weight Stage 1 and Stage 2 equally for classification, plus smoothing
    total_loss = loss_s1 + loss_s2 + (SMOOTHING_LOSS_WEIGHT * loss_smooth)

    return total_loss, loss_s1, loss_s2, loss_smooth


def train_epoch(model, loader, optimizer, criterion_ce, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        s1_logits, s2_logits = model(inputs)

        # Compute loss
        loss, l1, l2, l_smooth = compute_loss(
            s1_logits, s2_logits, targets, criterion_ce
        )

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Levenshtein distance.
    """
    model.eval()

    all_preds = []
    all_truths = []

    with torch.no_grad():
        for inputs, targets, sample_id in loader:
            inputs = inputs.to(device)
            # targets is (Batch, Time)

            # Forward pass
            _, s2_logits = model(inputs)

            # Use Stage 2 output for final prediction
            # s2_logits shape: (1, Time, NumClasses)

            # Decode predictions (batch size is 1 for validation)
            # Squeeze batch dim -> (Time, NumClasses)
            logits_seq = s2_logits.squeeze(0)
            pred_seq = decode_predictions(logits_seq)
            all_preds.append(pred_seq)

            # Decode ground truth
            # targets shape: (1, Time) -> squeeze -> (Time,)
            target_seq_raw = targets.squeeze(0).cpu().numpy()
            # We need to collapse the ground truth similarly to how we handle preds
            # (Run-Length Encoding + Remove Background) to get the list of gestures
            truth_seq = decode_predictions(target_seq_raw)
            all_truths.append(truth_seq)

    # Compute metric
    score = compute_levenshtein_ratio(all_preds, all_truths)
    return score


def train_model():
    """
    Main driver function to train the Cascaded Refinement Network.
    """
    # Reproducibility
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_data_loaders()

    # Model
    model = CascadedRefinementNet().to(device)

    # Loss Setup
    # Class weights: Background (0) gets BG_CLASS_WEIGHT, others get 1.0
    weights = torch.ones(NUM_CLASSES)
    weights[0] = BG_CLASS_WEIGHT
    weights = weights.to(device)

    criterion_ce = nn.CrossEntropyLoss(weight=weights)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training State
    best_val_score = float("inf")  # Levenshtein distance (lower is better)
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion_ce, device)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch}: Train Loss = {train_loss}, Val Levenshtein = {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_val_score}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_val_score}")
    return best_model_path
