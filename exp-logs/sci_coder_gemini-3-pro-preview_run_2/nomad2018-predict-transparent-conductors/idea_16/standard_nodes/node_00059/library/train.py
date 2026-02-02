import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, compute_rmsle
from library.data import CrystalGraphDataset, collate_graphs
from library.model import IR_CGCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (
        atom_fea,
        edge_index,
        edge_fea,
        batch_index,
        targets,
        _,
    ) in enumerate(loader):
        # Move data to device
        atom_fea = atom_fea.to(device)
        edge_index = edge_index.to(device)
        edge_fea = edge_fea.to(device)
        batch_index = batch_index.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(atom_fea, edge_index, edge_fea, batch_index)

        # Compute loss (MSE on standardized targets)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, scaler):
    """
    Evaluates the model on the validation set.
    Returns the average MSE loss (scaled) and the RMSLE (unscaled, original units).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for atom_fea, edge_index, edge_fea, batch_index, targets, _ in loader:
            # Move data to device
            atom_fea = atom_fea.to(device)
            edge_index = edge_index.to(device)
            edge_fea = edge_fea.to(device)
            batch_index = batch_index.to(device)
            targets_dev = targets.to(device)

            # Forward pass
            preds = model(atom_fea, edge_index, edge_fea, batch_index)

            # Compute loss on scaled data
            loss = criterion(preds, targets_dev)
            running_loss += loss.item() * targets.size(0)

            # Collect predictions and targets for RMSLE calculation
            # Move to CPU numpy
            preds_np = preds.cpu().numpy()
            targets_np = targets.numpy()  # targets from loader are already CPU tensors

            # Inverse transform to get original units (eV)
            preds_unscaled = scaler.inverse_transform(preds_np)
            targets_unscaled = scaler.inverse_transform(targets_np)

            all_preds.append(preds_unscaled)
            all_targets.append(targets_unscaled)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSLE metric
    rmsle = compute_rmsle(all_preds, all_targets)

    return epoch_loss, rmsle


def run_training():
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = CrystalGraphDataset(Config.TRAIN_METADATA_PATH, mode="train")
    val_dataset = CrystalGraphDataset(Config.VAL_METADATA_PATH, mode="val")

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = IR_CGCNN(Config).to(device)

    # 4. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation RMSLE stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Training Loop
    print("Starting Training...")
    best_val_rmsle = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmsle = validate(
            model, val_loader, criterion, device, train_dataset.scaler
        )

        # Scheduler step
        scheduler.step(val_rmsle)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val Loss (MSE): {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmsle": val_rmsle,
                },
                Config.BEST_MODEL_PATH,
            )
            print(f"  -> New best model saved! RMSLE: {val_rmsle:.10f}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference and Submission
    print("\nStarting Inference on Test Set...")

    # Load best model
    checkpoint = load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with Val RMSLE: {checkpoint['val_rmsle']:.10f}"
    )

    # Prepare Test Loader
    test_dataset = CrystalGraphDataset(Config.TEST_METADATA_PATH, mode="test")
    # Important: The test dataset needs the scaler from training to inverse transform predictions.
    # The dataset class handles loading the saved scaler automatically in 'test' mode
    # if Config.TARGET_SCALER_PATH exists (which it should after training).

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for atom_fea, edge_index, edge_fea, batch_index, _, ids in test_loader:
            atom_fea = atom_fea.to(device)
            edge_index = edge_index.to(device)
            edge_fea = edge_fea.to(device)
            batch_index = batch_index.to(device)

            # Predict
            preds = model(atom_fea, edge_index, edge_fea, batch_index)

            # Inverse transform
            preds_np = preds.cpu().numpy()
            preds_unscaled = test_dataset.scaler.inverse_transform(preds_np)

            # Store
            ids_list.extend(ids.numpy())
            preds_list.append(preds_unscaled)

    # Concatenate predictions
    all_preds = np.concatenate(preds_list, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Ensure non-negative predictions (physically required)
    submission_df["formation_energy_ev_natom"] = submission_df[
        "formation_energy_ev_natom"
    ].clip(lower=0.0)
    submission_df["bandgap_energy_ev"] = submission_df["bandgap_energy_ev"].clip(
        lower=0.0
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Done.")
