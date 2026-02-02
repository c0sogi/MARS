import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.model import CR_WDS
from library.data import get_dataloaders
from library.utils import inverse_log_transform, compute_rmsle


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, and prediction processes.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move batch data to device
            atomic_features = batch["atomic_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            global_features = batch["global_features"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            batch_dict = {
                "atomic_features": atomic_features,
                "mask": mask,
                "global_features": global_features,
            }
            outputs = self.model(batch_dict)

            # Compute loss (MSE on log-transformed targets)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                mask = batch["mask"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                targets = batch["targets"].to(self.device)

                batch_dict = {
                    "atomic_features": atomic_features,
                    "mask": mask,
                    "global_features": global_features,
                }
                outputs = self.model(batch_dict)

                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * targets.size(0)

                # Collect for RMSLE calculation (inverse transform first)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        # Concatenate and inverse transform
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        pred_orig = inverse_log_transform(all_preds)
        target_orig = inverse_log_transform(all_targets)

        # Compute RMSLE on original scale
        # Note: compute_rmsle handles clipping negative values internally
        rmsle = compute_rmsle(target_orig, pred_orig)

        return val_loss, rmsle

    def fit(self, train_loader, val_loader, epochs, patience):
        best_rmsle = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_rmsle = self.validate(val_loader)

            # Step scheduler based on validation loss (MSE in log space)
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val RMSLE: {val_rmsle:.6f}"
            )

            # Early Stopping Check
            if val_rmsle < best_rmsle:
                best_rmsle = val_rmsle
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # print(f"  -> New best model saved! RMSLE: {best_rmsle:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Validation RMSLE: {best_rmsle:.6f}")

    def predict(self, test_loader):
        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                mask = batch["mask"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                ids = batch["ids"]

                batch_dict = {
                    "atomic_features": atomic_features,
                    "mask": mask,
                    "global_features": global_features,
                }
                outputs = self.model(batch_dict)

                all_preds.append(outputs.cpu().numpy())
                all_ids.extend(ids)

        # Concatenate and inverse transform
        all_preds = np.concatenate(all_preds, axis=0)
        pred_orig = inverse_log_transform(all_preds)

        # Ensure non-negative predictions for physical validity
        pred_orig = np.maximum(pred_orig, 0.0)

        return all_ids, pred_orig


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # get_dataloaders handles caching internally
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # 3. Model Initialization
    model = CR_WDS()

    # 4. Training
    trainer = Trainer(model, device)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 5. Prediction
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    ids, predictions = trainer.predict(test_loader)

    # 6. Submission Generation
    print(f"Generating submission file at {Config.SUBMISSION_PATH}...")

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to match sample submission structure usually
    submission_df.sort_values("id", inplace=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    # This block is for local testing only, the competition runner imports the module
    # run_training()
    pass
