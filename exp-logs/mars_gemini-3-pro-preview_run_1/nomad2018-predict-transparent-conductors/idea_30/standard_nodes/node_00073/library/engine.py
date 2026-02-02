import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    FACTOR,
    MIN_LR,
    SEED,
)
from library.model import GPA_WDS
from library.data import get_dataloaders
from library.utils import inverse_log_transform_targets, calculate_rmsle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Engine:
    def __init__(self, device):
        self.device = device
        self.model = GPA_WDS().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # Scheduler patience is typically less than early stopping patience
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=FACTOR,
            patience=max(2, PATIENCE // 2),
            min_lr=MIN_LR,
        )
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for (
            batch_atomic,
            batch_indices,
            batch_global,
            batch_targets,
            batch_ids,
        ) in train_loader:
            # Move data to device
            batch_atomic = batch_atomic.to(self.device)
            batch_indices = batch_indices.to(self.device)
            batch_global = batch_global.to(self.device)
            batch_targets = batch_targets.to(self.device)
            # batch_ids is not used in forward/loss, just metadata

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_atomic, batch_global, batch_indices, batch_ids)

            # Calculate loss (MSE on log-transformed targets)
            loss = self.criterion(outputs, batch_targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_global.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for (
                batch_atomic,
                batch_indices,
                batch_global,
                batch_targets,
                batch_ids,
            ) in val_loader:
                batch_atomic = batch_atomic.to(self.device)
                batch_indices = batch_indices.to(self.device)
                batch_global = batch_global.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(
                    batch_atomic, batch_global, batch_indices, batch_ids
                )
                loss = self.criterion(outputs, batch_targets)

                running_loss += loss.item() * batch_global.size(0)

                # Store for metric calculation
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)

        # Concatenate all batches
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Inverse transform to get original scale
        all_preds_orig = inverse_log_transform_targets(all_preds)
        all_targets_orig = inverse_log_transform_targets(all_targets)

        # Calculate RMSLE
        rmsle = calculate_rmsle(all_targets_orig, all_preds_orig)

        return epoch_loss, rmsle

    def run_training(self, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_rmsle = self.evaluate(val_loader)

            # Step scheduler based on validation loss
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val RMSLE: {val_rmsle:.6f} - "
                f"LR: {current_lr:.2e}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"  New best model saved! (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    def generate_submission(self, test_loader):
        print("Generating submission...")

        # Load best model
        if os.path.exists(MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model from checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch_atomic, batch_indices, batch_global, _, batch_ids in test_loader:
                batch_atomic = batch_atomic.to(self.device)
                batch_indices = batch_indices.to(self.device)
                batch_global = batch_global.to(self.device)

                # Forward pass
                outputs = self.model(
                    batch_atomic, batch_global, batch_indices, batch_ids
                )

                # Inverse transform predictions
                preds_orig = inverse_log_transform_targets(outputs.cpu().numpy())
                ids = batch_ids.numpy()

                for i in range(len(ids)):
                    results.append(
                        {
                            "id": int(ids[i]),
                            "formation_energy_ev_natom": preds_orig[i, 0],
                            "bandgap_energy_ev": preds_orig[i, 1],
                        }
                    )

        # Create DataFrame and save
        submission_df = pd.DataFrame(results)
        # Ensure correct column order
        submission_df = submission_df[
            ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        ]
        submission_df.sort_values("id", inplace=True)

        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")


def run_engine(load_cached_data=True):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Engine
    engine = Engine(device)

    # Train
    engine.run_training(train_loader, val_loader)

    # Generate Submission
    engine.generate_submission(test_loader)
