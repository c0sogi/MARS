import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.dataset import get_dataloader
from library.loss import MaskedMCRMSELoss
from library.model import EIPFN


def train_model(debug=False, epochs=None, batch_size=None):
    """
    Trains the EIPFN model using the Embedded-Input Pure-Feedback strategy.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
        epochs (int, optional): Number of training epochs. Defaults to Config.EPOCHS.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if epochs is None:
        epochs = Config.EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # 2. Data Loaders
    print(f"Initializing DataLoaders (Debug={debug})...")
    train_loader = get_dataloader(
        mode="train",
        load_cached_data=True,
        debug=debug,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = get_dataloader(
        mode="val",
        load_cached_data=True,
        debug=debug,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Model & Optimization
    print("Initializing Model...")
    model = EIPFN().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )
    criterion = MaskedMCRMSELoss()

    # 4. Training Loop
    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        # --- Training Phase ---
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Pass 1: Zero feedback
            # y_prev is None inside model handles the zero initialization
            preds_1 = model(inputs, partner_indices, y_prev=None)
            loss_1 = criterion(preds_1, targets)

            # Pass 2: Feedback from Pass 1 (Detached)
            # We detach preds_1 to stop gradients from flowing back through the feedback loop into Pass 1
            preds_1_detached = preds_1.detach()
            preds_2 = model(inputs, partner_indices, y_prev=preds_1_detached)
            loss_2 = criterion(preds_2, targets)

            # Combined Loss
            loss = loss_2 + 0.5 * loss_1

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_metric = MCRMSE(scored_indices=Config.TARGET_INDICES)

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                targets = batch["targets"].to(device)

                # Create mask for validation metric (only first 68 positions)
                B, L = inputs.shape[:2]
                mask = torch.zeros((B, L), device=device)
                mask[:, : Config.SEQ_SCORED] = 1.0

                # Inference Pass 1
                preds_1 = model(inputs, partner_indices, y_prev=None)

                # Inference Pass 2 (Feedback)
                preds_2 = model(inputs, partner_indices, y_prev=preds_1)

                # Update Global Metric
                val_metric.update(preds_2, targets, mask)

        val_score = val_metric.compute()

        # --- Logging & Checkpointing ---
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.8f} | "
            f"Val MCRMSE: {val_score:.16f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        scheduler.step(val_score)

        if val_score < best_val_score:
            print(
                f"Validation score improved ({best_val_score:.8f} -> {val_score:.8f}). Saving model..."
            )
            best_val_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation MCRMSE: {best_val_score:.16f}")
