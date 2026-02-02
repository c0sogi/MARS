import os
import time
import numpy as np
import torch
import torch.optim as optim

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    BEST_MODEL_PATH,
    SCORED_SEQ_LENGTH,
    SCORED_INDICES,
    NUM_WORKERS,
)
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import RIS_DRN, loss_fn


def train_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        partner_map = batch["partner_map"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass (Returns both passes for iterative refinement loss)
        logits_1, logits_2 = model(inputs, partner_map)

        # Calculate loss (Strict masking is handled inside loss_fn)
        loss = loss_fn(logits_1, logits_2, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using global MCRMSE.
    Collecting all predictions first ensures correct global statistics.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_map = batch["partner_map"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            # We only care about the final refined output (logits_2) for validation
            _, logits_2 = model(inputs, partner_map)

            # Slice to valid scored sequence length (0-67)
            # Shape: (Batch, 68, 5)
            preds_sliced = logits_2[:, :SCORED_SEQ_LENGTH, :]
            targets_sliced = targets[:, :SCORED_SEQ_LENGTH, :]

            # Move to CPU and collect
            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets_sliced.cpu().numpy())

    # Concatenate to form global arrays: (N_total, 68, 5)
    global_preds = np.concatenate(all_preds, axis=0)
    global_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE using the utility function
    # It handles column filtering based on SCORED_INDICES internally if passed,
    # but we can also pass the full arrays and let it filter.
    score = mcrmse(global_targets, global_preds, scored_indices=SCORED_INDICES)

    return score


def run_training():
    """
    Main function to setup and run the training pipeline.
    """
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    # load_cached_data=True allows using pre-processed .npz files if they exist
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = RIS_DRN().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_score})")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_score}")


if __name__ == "__main__":
    run_training()
