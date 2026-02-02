import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mcrmse_loss, metric_mcrmse_scored
from library.data import get_dataloaders
from library.model import RNAModel


def train_model(config=Config, load_cached_data=True, save_path=None):
    """
    Executes the training pipeline for the High-Capacity GLU-Decoupled BiGRU model.

    Args:
        config: Configuration class containing hyperparameters.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        save_path (str): Path to save the best model checkpoint.
                         Defaults to Config.CACHE_DIR/best_model.pth.

    Returns:
        model: The trained model with the best weights loaded.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_path is None:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        save_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    model = RNAModel(config).to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            p_idx = batch["pair_indices"].to(device)
            p_mask = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(features, p_idx, p_mask)

            # Slice to scored length (68) for loss calculation
            # Ground truth is only valid for the first 68 bases.
            preds_scored = preds[:, : config.SEQ_SCORED, :]
            targets_scored = targets[:, : config.SEQ_SCORED, :]

            # Calculate Loss (MCRMSE on all 5 columns)
            loss = mcrmse_loss(preds_scored, targets_scored)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                p_idx = batch["pair_indices"].to(device)
                p_mask = batch["pair_masks"].to(device)
                targets = batch["targets"].to(device)

                preds = model(features, p_idx, p_mask)

                all_preds.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate Competition Metric
        # metric_mcrmse_scored handles slicing and column selection internally
        val_metric = metric_mcrmse_scored(all_preds, all_targets, config.SEQ_SCORED)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss} | Val MCRMSE: {val_metric}"
        )

        # Checkpointing & Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_metric}")

    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
