import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch_geometric.loader import DataLoader

from library.config import Config
from library.model import MSR_CGCNN
from library.data import get_dataset
from library.utils import (
    set_seed,
    save_checkpoint,
    compute_rmsle,
    AverageMeter,
    StandardScaler,
)


class Trainer:
    """
    Handles the training and validation loop for the MSR-CGCNN model.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, device, target_scaler
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.target_scaler = target_scaler
        # MSE Loss is appropriate for regression on standardized targets
        self.criterion = nn.MSELoss()

    def train_epoch(self):
        """
        Performs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch)

            # Compute loss
            loss = self.criterion(preds, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), batch.num_graphs)

        return loss_meter.avg

    def validate(self):
        """
        Evaluates the model on the validation set.
        Computes MSE (on scaled data) and RMSLE (on original scale).
        """
        self.model.eval()
        loss_meter = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass
                preds = self.model(batch)

                # Compute loss (on scaled targets)
                loss = self.criterion(preds, batch.y)
                loss_meter.update(loss.item(), batch.num_graphs)

                # Collect predictions and targets for metric calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Inverse transform to get real units (eV/atom and eV)
        real_preds = self.target_scaler.inverse_transform(all_preds)
        real_targets = self.target_scaler.inverse_transform(all_targets)

        # Compute RMSLE column-wise
        # Column 0: formation_energy, Column 1: bandgap_energy
        rmsle_formation = compute_rmsle(real_targets[:, 0], real_preds[:, 0])
        rmsle_bandgap = compute_rmsle(real_targets[:, 1], real_preds[:, 1])

        # Average RMSLE (or sum, depending on specific competition needs, usually mean)
        mean_rmsle = (rmsle_formation + rmsle_bandgap) / 2.0

        metrics = {
            "val_loss": loss_meter.avg,
            "rmsle_formation": rmsle_formation,
            "rmsle_bandgap": rmsle_bandgap,
            "mean_rmsle": mean_rmsle,
        }

        return metrics


def train_model(
    max_epochs=Config.MAX_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    debug=Config.DEBUG,
):
    """
    Main function to set up and run the training process.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_dataset handles loading metadata, processing graphs, caching, and scaling.
    print("Preparing datasets...")
    train_dataset = get_dataset("train", load_cached_data=True, debug=debug)
    val_dataset = get_dataset("val", load_cached_data=True, debug=debug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load target scaler for inverse transformation during validation
    # This file is created by get_dataset('train')
    target_scaler = StandardScaler()
    if os.path.exists(Config.TARGET_SCALER_PATH):
        target_scaler.load_state_dict(
            torch.load(Config.TARGET_SCALER_PATH, weights_only=False)
        )
    else:
        # Fallback for safety, though this implies training data wasn't processed correctly
        print(
            "Warning: Target scaler not found. Validation metrics will be calculated on scaled data."
        )
        target_scaler.mean = np.array([0.0, 0.0])
        target_scaler.std = np.array([1.0, 1.0])

    # 3. Model Initialization
    model = MSR_CGCNN(config=Config).to(device)

    # 4. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        target_scaler=target_scaler,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, max_epochs + 1):
        train_loss = trainer.train_epoch()
        val_metrics = trainer.validate()

        val_loss = val_metrics["val_loss"]
        mean_rmsle = val_metrics["mean_rmsle"]

        print(
            f"Epoch {epoch}: "
            f"Train Loss (MSE): {train_loss}, "
            f"Val Loss (MSE): {val_loss}, "
            f"Val Mean RMSLE: {mean_rmsle}"
        )

        # Checkpoint and Early Stopping
        # We monitor MSE Loss for stability, though RMSLE is the final metric
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved at epoch {epoch} with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("Training complete.")
    return model
