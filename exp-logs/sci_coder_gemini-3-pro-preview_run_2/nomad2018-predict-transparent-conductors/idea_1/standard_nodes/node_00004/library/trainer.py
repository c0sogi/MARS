import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.model import CGCNN
from library.data_loader import get_train_val_test_loaders
from library.utils import AverageMeter, Normalizer, set_seed


class Trainer:
    """
    Trainer class for the Crystal Graph Convolutional Neural Network.
    Manages training, validation, and prediction processes.
    """

    def __init__(self, model, optimizer, criterion, scheduler, device, normalizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.normalizer = normalizer

    def train_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (
            atom_fea,
            edge_index,
            edge_dist,
            batch_index,
            targets,
            ids,
        ) in enumerate(train_loader):
            # Move data to device
            atom_fea = atom_fea.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_dist = edge_dist.to(self.device)
            batch_index = batch_index.to(self.device)
            targets = targets.to(self.device)

            # Normalize targets for training stability
            targets_norm = self.normalizer.norm(targets)

            # Forward pass
            preds = self.model(atom_fea, edge_index, edge_dist, batch_index)

            # Compute loss
            loss = self.criterion(preds, targets_norm)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), targets.size(0))

        return losses.avg

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average MSE loss (normalized) and RMSLE (denormalized).
        """
        self.model.eval()
        losses = AverageMeter()
        rmsle_meter = AverageMeter()

        with torch.no_grad():
            for (
                atom_fea,
                edge_index,
                edge_dist,
                batch_index,
                targets,
                ids,
            ) in val_loader:
                atom_fea = atom_fea.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_dist = edge_dist.to(self.device)
                batch_index = batch_index.to(self.device)
                targets = targets.to(self.device)

                # Predict
                preds_norm = self.model(atom_fea, edge_index, edge_dist, batch_index)

                # Loss on normalized targets (consistent with training for early stopping)
                targets_norm = self.normalizer.norm(targets)
                loss = self.criterion(preds_norm, targets_norm)
                losses.update(loss.item(), targets.size(0))

                # Calculate RMSLE on denormalized (real) values
                preds = self.normalizer.denorm(preds_norm)

                # Ensure non-negative for log (physics constraint: energies >= 0 approx)
                preds_clamped = torch.clamp(preds, min=0.0)
                targets_clamped = torch.clamp(targets, min=0.0)

                # Column-wise RMSLE calculation
                # RMSLE = sqrt(mean((log(p+1) - log(t+1))^2))
                log_diff = torch.log1p(preds_clamped) - torch.log1p(targets_clamped)
                mse_log = torch.mean(
                    log_diff**2, dim=0
                )  # Mean over batch for each column
                rmsle_cols = torch.sqrt(mse_log)
                rmsle_val = torch.mean(rmsle_cols).item()  # Mean over the two targets

                rmsle_meter.update(rmsle_val, targets.size(0))

        return losses.avg, rmsle_meter.avg

    def predict(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        ids_list = []
        preds_list = []

        print("Generating predictions...")
        with torch.no_grad():
            for (
                atom_fea,
                edge_index,
                edge_dist,
                batch_index,
                targets,
                ids,
            ) in test_loader:
                atom_fea = atom_fea.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_dist = edge_dist.to(self.device)
                batch_index = batch_index.to(self.device)

                preds_norm = self.model(atom_fea, edge_index, edge_dist, batch_index)
                preds = self.normalizer.denorm(preds_norm)

                ids_list.append(ids.cpu())
                preds_list.append(preds.cpu())

        all_ids = torch.cat(ids_list).numpy().flatten()
        all_preds = torch.cat(preds_list).numpy()

        # Clamp negative predictions to 0 as physical quantities cannot be negative
        all_preds = np.maximum(all_preds, 0.0)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "id": all_ids,
                "formation_energy_ev_natom": all_preds[:, 0],
                "bandgap_energy_ev": all_preds[:, 1],
            }
        )

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


def run_training(
    epochs=100,
    batch_size=64,
    lr=2e-3,
    weight_decay=1e-5,
    patience=15,
    radius=5.0,
    load_cached_data=True,
    num_workers=2,
):
    """
    Main function to setup and run the training pipeline.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        batch_size=batch_size,
        radius=radius,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # 2. Calculate Normalization Statistics from Training Data
    print("Calculating normalizer statistics from training data...")
    all_targets = []
    # Iterate once to collect all targets
    for _, _, _, _, targets, _ in train_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)

    # Initialize Normalizer with training stats
    normalizer = Normalizer(tensor=all_targets)
    # Move normalizer mean/std to device for efficient denorm during training
    normalizer.mean = normalizer.mean.to(device)
    normalizer.std = normalizer.std.to(device)

    print(f"Normalizer Mean: {normalizer.mean.cpu().numpy()}")
    print(f"Normalizer Std:  {normalizer.std.cpu().numpy()}")

    # 3. Model Setup
    # orig_atom_fea_len=4 (Al, Ga, In, O) mapped to 0,1,2,3
    model = CGCNN(
        orig_atom_fea_len=4,
        atom_fea_len=64,
        n_conv=4,
        h_fea_len=128,
        n_h=2,
        n_targets=2,
        radius=radius,
        n_rbf=50,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    # Scheduler: Reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    trainer = Trainer(model, optimizer, criterion, scheduler, device, normalizer)

    # 4. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    # Directory for checkpoints
    ckpt_dir = "./working/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss, val_rmsle = trainer.validate(val_loader)

        # Step scheduler
        trainer.scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}: Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Val RMSLE: {val_rmsle:.8f} | Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # 5. Prediction
    print(f"Loading best model from {best_model_path} for prediction...")
    model.load_state_dict(torch.load(best_model_path))
    trainer.predict(test_loader, "./submission/submission.csv")
