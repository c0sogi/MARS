import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_log_mae
from library.model import DistanceWeightedGCN
from library.data import get_dataloaders


class Engine:
    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # Initialize Model
        self.model = DistanceWeightedGCN().to(self.device)

        # Loss Function (L1 Loss for MAE)
        self.criterion = nn.L1Loss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch)

            # Targets are normalized in the dataset
            targets = batch.y

            # Ensure shapes match: preds [N, 1] -> [N], targets [N]
            loss = self.criterion(preds.squeeze(-1), targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch.num_graphs

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                preds = self.model(batch)
                targets = batch.y

                # Calculate loss on normalized data (for scheduler/monitoring)
                loss = self.criterion(preds.squeeze(-1), targets)
                running_loss += loss.item() * batch.num_graphs

                # Store for metric calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_types.append(batch.couple_type.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        # Concatenate arrays
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        all_types = np.concatenate(all_types).flatten()

        # Denormalize to calculate physical metric
        # y_raw = y_norm * std + mean
        preds_raw = all_preds * Config.TARGET_STD + Config.TARGET_MEAN
        targets_raw = all_targets * Config.TARGET_STD + Config.TARGET_MEAN

        # Calculate Log MAE
        metric = calculate_log_mae(preds_raw, targets_raw, all_types)

        return val_loss, metric

    def run(self):
        print(f"Starting training on device: {self.device}")

        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_metric = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val LogMAE: {val_metric:.15f}"
            )

            # Early Stopping and Model Saving
            if val_metric < best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved! LogMAE: {best_metric:.15f}")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("-" * 30)
        print(f"Training complete. Best LogMAE: {best_metric:.15f}")

        # Generate Submission
        self.predict_and_submit(test_loader)

    def predict_and_submit(self, test_loader):
        print("Loading best model for inference...")
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("Error: Model file not found. Cannot generate submission.")
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                preds = self.model(batch)

                all_preds.append(preds.cpu().numpy())
                all_ids.append(batch.id.cpu().numpy())

        # Concatenate
        if len(all_preds) == 0:
            print("Warning: No predictions generated. Test loader might be empty.")
            return

        all_preds = np.concatenate(all_preds).flatten()
        all_ids = np.concatenate(all_ids).flatten()

        # Denormalize predictions
        preds_raw = all_preds * Config.TARGET_STD + Config.TARGET_MEAN

        # Create DataFrame
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": preds_raw})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save submission
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
