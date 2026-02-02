import os
import torch
import numpy as np
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.utils import mcrmse_metric


def train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Executes the training pipeline for the RNA degradation prediction model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = RNAModel(config=Config)
    model.to(device)

    # 4. Optimization & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = MaskedMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move data to device
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(seq, loop, dist)

            # Calculate loss (MaskedMSELoss handles slicing internally)
            loss = criterion(outputs, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_sum / num_batches

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        # Store predictions and targets for metric calculation
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                target = batch["target"].to(device)

                outputs = model(seq, loop, dist)

                loss = criterion(outputs, target)
                val_loss_sum += loss.item()
                val_batches += 1

                # Collect data for MCRMSE calculation
                # We must slice to the scored length (68) for the metric
                pred_slice = outputs[:, : Config.PRED_LEN, :].cpu().numpy()
                target_slice = target[:, : Config.PRED_LEN, :].cpu().numpy()

                all_preds.append(pred_slice)
                all_targets.append(target_slice)

        avg_val_loss = val_loss_sum / val_batches

        # Concatenate all batches
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Calculate Metric
        val_mcrmse = mcrmse_metric(y_true, y_pred)

        # Step Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            print(
                f"New best model found! (Previous: {best_mcrmse}, Current: {val_mcrmse})"
            )
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"Model saved to {Config.MODEL_PATH}")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")
