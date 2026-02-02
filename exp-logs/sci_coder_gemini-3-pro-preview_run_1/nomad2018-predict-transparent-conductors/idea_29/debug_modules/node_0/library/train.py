import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import time

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    WORKING_DIR,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    ATOMIC_HIDDEN_DIM,
    GLOBAL_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    DROPOUT,
)
from library.dataset import MaterialDataset, collate_materials
from library.model import PIGWDS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    max_samples=None,
    load_cached_data=True,
):
    """
    Trains the PIG-WDS model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.
        max_samples (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Initializing Training Dataset...")
    # Train dataset computes scalers automatically
    train_dataset = MaterialDataset(
        metadata_path=TRAIN_CSV,
        scalers=None,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # Save scalers for inference
    scalers_path = os.path.join(WORKING_DIR, "scalers.npz")
    np.savez(scalers_path, **train_dataset.scalers)
    print(f"Scalers saved to {scalers_path}")

    print("Initializing Validation Dataset...")
    # Val dataset uses training scalers
    val_dataset = MaterialDataset(
        metadata_path=VAL_CSV,
        scalers=train_dataset.scalers,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 2. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_materials,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Initialization
    # Infer input dimensions from a sample
    sample = train_dataset[0]
    atomic_input_dim = sample["atomic_features"].shape[1]
    global_input_dim = sample["global_features"].shape[0]
    output_dim = sample["target"].shape[0]

    print(f"Atomic Input Dim: {atomic_input_dim}")
    print(f"Global Input Dim: {global_input_dim}")
    print(f"Output Dim: {output_dim}")

    model = PIGWDS(
        atomic_input_dim=atomic_input_dim,
        global_input_dim=global_input_dim,
        atomic_hidden_dim=ATOMIC_HIDDEN_DIM,
        global_hidden_dim=GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        output_dim=output_dim,
        dropout=DROPOUT,
    ).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pt")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Training ---
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            # Move data to device
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(atomic_feats, global_feats, mask)

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * targets.size(0)

        avg_train_loss = train_loss_sum / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_feats = batch["atomic_features"].to(device)
                global_feats = batch["global_features"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["targets"].to(device)

                predictions = model(atomic_feats, global_feats, mask)
                loss = criterion(predictions, targets)

                val_loss_sum += loss.item() * targets.size(0)

        avg_val_loss = val_loss_sum / len(val_dataset)

        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Metrics (RMSLE approximation since targets are log-transformed)
        train_rmsle = np.sqrt(avg_train_loss)
        val_rmsle = np.sqrt(avg_val_loss)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train MSE: {avg_train_loss:.8f} | "
            f"Val MSE: {avg_val_loss:.8f} | "
            f"Train RMSLE: {train_rmsle:.8f} | "
            f"Val RMSLE: {val_rmsle:.8f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping and Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MSE: {best_val_loss:.8f}")
