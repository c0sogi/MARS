import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    get_device,
    calculate_weighted_log_loss,
)
from library.dataset import CervicalSpineDataset
from library.model import CervicalMILModel
from library.loss import ImplicitlyWeightedMultiTaskLoss

logger = get_logger("trainer")


class Trainer:
    def __init__(self, debug=False):
        self.device = get_device()
        self.debug = debug
        self.best_score = float("inf")

        # Load Metadata
        self.train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        self.val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        if self.debug:
            logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            self.train_df = self.train_df.head(Config.DEBUG_SAMPLE_SIZE)
            self.val_df = self.val_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Datasets & Loaders
        self.train_dataset = CervicalSpineDataset(self.train_df, mode="train")
        self.val_dataset = CervicalSpineDataset(self.val_df, mode="val")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Model
        logger.info(f"Initializing model: {Config.MODEL_NAME}")
        self.model = CervicalMILModel(num_classes=Config.NUM_CLASSES, pretrained=True)
        self.model.to(self.device)

        # Loss
        self.criterion = ImplicitlyWeightedMultiTaskLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Decoupled Cosine Annealing)
        # T_max is set to a multiple of total epochs to prevent premature decay to 0
        t_max = int(Config.EPOCHS * Config.T_MAX_MULTIPLIER)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=1e-6)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # Loss calculation
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        # Containers for metric calculation
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * images.size(0)
                dataset_size += images.size(0)

                # Collect probabilities for metric
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = running_loss / dataset_size

        # Prepare DataFrames for weighted log loss calculation
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        y_true = pd.DataFrame(all_targets, columns=cols)
        y_pred = pd.DataFrame(all_preds, columns=cols)

        # Calculate competition metric
        metric_score = calculate_weighted_log_loss(y_true, y_pred)

        return val_loss, metric_score

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def fit(self, epochs=Config.EPOCHS, patience=5):
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")

        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metric = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_metric}"
            )

            # Early Stopping & Checkpointing
            # We optimize for the competition metric (Weighted Log Loss) - lower is better
            if val_metric < self.best_score:
                self.best_score = val_metric
                self.save_model(best_model_path)
                patience_counter = 0
                logger.info(f"New best score: {self.best_score}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Metric Score: {self.best_score}")


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Main entry point to run the training pipeline.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    trainer = Trainer(debug=debug)
    trainer.fit(epochs=epochs)
