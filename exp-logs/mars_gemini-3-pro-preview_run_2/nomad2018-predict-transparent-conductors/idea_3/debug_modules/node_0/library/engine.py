import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.data import get_data, CrystalGraphDataset, collate_graphs
from library.model import DBGT
from library.utils import setup_logger, compute_rmsle, TargetScaler


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch to device
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        targets = batch["y"].to(device)

        # Transform targets
        targets_scaled = scaler.transform(targets)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch": batch_idx,
            }
        )

        # Compute loss
        loss = criterion(outputs, targets_scaled)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, scaler, device):
    """
    Evaluates the model on the validation set.
    Returns the average MSE loss (scaled) and RMSLE (original scale).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_idx = batch["batch"].to(device)
            targets = batch["y"].to(device)

            # Forward pass
            outputs = model(
                {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "batch": batch_idx,
                }
            )

            # Compute Loss on scaled targets for consistency with training
            targets_scaled = scaler.transform(targets)
            loss = criterion(outputs, targets_scaled)
            running_loss += loss.item() * targets.size(0)

            # Inverse transform for metric calculation
            preds_orig = scaler.inverse_transform(outputs)

            all_preds.append(preds_orig.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSLE
    rmsle = compute_rmsle(all_targets, all_preds)

    return epoch_loss, rmsle


def run_training(load_cached_data=True):
    """
    Main execution function for training the DB-GT model.
    """
    logger = setup_logger(log_file=os.path.join(Config.WORKING_DIR, "training.log"))
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading Data...")
    train_graphs, train_targets, train_ids = get_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )
    val_graphs, val_targets, val_ids = get_data(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )

    # 2. Initialize Scaler
    logger.info("Initializing and fitting TargetScaler...")
    scaler = TargetScaler()
    scaler.fit(np.array(train_targets))
    logger.info(f"Targets Mean: {scaler.mean}, Std: {scaler.std}")

    # 3. Create Datasets and Dataloaders
    train_dataset = CrystalGraphDataset(train_graphs, train_targets, train_ids)
    val_dataset = CrystalGraphDataset(val_graphs, val_targets, val_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Initialize Model
    logger.info("Initializing Model...")
    model = DBGT(config=Config).to(device)

    # 5. Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 6. Training Loop with Early Stopping
    logger.info("Starting Training...")
    best_val_rmsle = float("inf")
    patience_counter = 0

    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, scaler, device)

        logger.info(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val RMSLE: {val_rmsle:.10f}"
        )

        # Early Stopping Check
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_rmsle": val_rmsle,
                "config": Config.__dict__,
            }
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Training complete. Best Val RMSLE: {best_val_rmsle}")


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    """
    logger = setup_logger(log_file=os.path.join(Config.WORKING_DIR, "submission.log"))
    device = torch.device(Config.DEVICE)

    # 1. Load Test Data
    logger.info("Loading Test Data...")
    test_graphs, test_targets, test_ids = get_data(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )

    test_dataset = CrystalGraphDataset(test_graphs, test_targets, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model and Scaler
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")
    if not os.path.exists(best_model_path):
        logger.error(f"Checkpoint not found at {best_model_path}. Run training first.")
        return

    logger.info(f"Loading checkpoint from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device)

    model = DBGT(config=Config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = TargetScaler()
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 3. Inference
    logger.info("Running Inference...")
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_idx = batch["batch"].to(device)
            ids = batch["id"]

            outputs = model(
                {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "batch": batch_idx,
                }
            )

            # Inverse transform
            preds_orig = scaler.inverse_transform(outputs)

            all_ids.extend(ids)
            all_preds.append(preds_orig.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # 4. Save Submission
    logger.info("Saving Submission...")
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Ensure non-negative predictions if physically required (Energy usually is, Bandgap is)
    # The metric function clips them, so we should probably clip them in submission too
    submission_df["formation_energy_ev_natom"] = submission_df[
        "formation_energy_ev_natom"
    ].clip(lower=0.0)
    submission_df["bandgap_energy_ev"] = submission_df["bandgap_energy_ev"].clip(
        lower=0.0
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
