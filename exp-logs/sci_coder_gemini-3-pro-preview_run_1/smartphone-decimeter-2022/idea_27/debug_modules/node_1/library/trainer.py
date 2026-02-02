import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import ResUNet1D
from library.loss import DeepSupervisionMAELoss
from library.data_processing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset, gnss_collate_fn
from library.utils import setup_logger


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "train.log"))

        # Initialize Model
        self.model = ResUNet1D().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Initialize Loss
        self.criterion = DeepSupervisionMAELoss().to(self.device)

        self.best_val_loss = float("inf")

    def get_dataloaders(self, load_cached_data=True):
        """
        Loads processed data and creates DataLoaders.
        """
        preprocessor = GNSSPreprocessor()

        # Load Dataframes
        train_df = preprocessor.process_train_data(load_cached_data=load_cached_data)
        val_df = preprocessor.process_val_data(load_cached_data=load_cached_data)

        # Debugging: Sample subset
        if Config.DEBUG:
            self.logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Create Datasets
        train_dataset = GnssSequenceDataset(train_df, is_test=False)
        val_dataset = GnssSequenceDataset(val_df, is_test=False)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=gnss_collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=gnss_collate_fn,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Prepare inputs
            # features: (B, T, C) -> Model expects (B, C, T)
            features = batch["features"].to(self.device).transpose(1, 2)
            targets = batch["targets"].to(self.device)  # (B, T, C_out)
            mask = batch["mask"].to(self.device)  # (B, T)

            self.optimizer.zero_grad()

            # Forward pass
            # final_out: (B, C_out, T), aux_outs: list of (B, C_out, T_aux)
            final_out, aux_outs = self.model(features)

            # Transpose outputs for loss calculation: (B, C, T) -> (B, T, C)
            final_out = final_out.transpose(1, 2)
            aux_outs_T = [aux.transpose(1, 2) for aux in aux_outs]

            # Compute Loss
            loss = self.criterion((final_out, aux_outs_T), targets, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        total_distance_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device).transpose(1, 2)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                final_out, aux_outs = self.model(features)

                # Transpose for loss
                final_out = final_out.transpose(1, 2)
                aux_outs_T = [aux.transpose(1, 2) for aux in aux_outs]

                # Loss
                loss = self.criterion((final_out, aux_outs_T), targets, mask)
                running_loss += loss.item()

                # Calculate Distance Error Metric (Meters)
                # final_out: (B, T, 2) -> (dN, dE)
                # target: (B, T, 2)
                # mask: (B, T)

                # Truncate to match length if necessary (though loss handles it, we need alignment for metric)
                seq_len = final_out.size(1)
                targets_trunc = targets[:, :seq_len, :]
                mask_trunc = mask[:, :seq_len]

                diff = final_out - targets_trunc
                dist_sq = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
                dist = torch.sqrt(dist_sq)  # (B, T)

                # Apply mask
                valid_dist = dist[mask_trunc]

                if valid_dist.numel() > 0:
                    total_distance_error += valid_dist.sum().item()
                    total_samples += valid_dist.numel()

        avg_loss = running_loss / len(dataloader)
        avg_dist_error = (
            total_distance_error / total_samples if total_samples > 0 else 0.0
        )

        return avg_loss, avg_dist_error

    def fit(self, load_cached_data=True):
        self.logger.info("Starting training pipeline...")

        train_loader, val_loader = self.get_dataloaders(load_cached_data)
        self.logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_dist_error = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {duration:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Mean Dist Err: {val_dist_error:.4f}m"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.logger.info(
                    f"  -> Model saved! Best Val Loss: {self.best_val_loss:.6f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    self.logger.info("  -> Early stopping triggered.")
                    break

        self.logger.info("Training complete.")
