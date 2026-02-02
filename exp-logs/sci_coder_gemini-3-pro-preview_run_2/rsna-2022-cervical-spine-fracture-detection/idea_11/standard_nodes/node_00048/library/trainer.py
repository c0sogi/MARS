import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import CervicalFractureNet
from library.utils import seed_everything, weighted_log_loss


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        # Create directory for saving models
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def get_dataloaders(self, debug=False):
        """
        Initializes datasets and dataloaders.
        """
        train_dataset = CervicalSpineDataset(
            metadata_path=Config.TRAIN_METADATA, mode="train", load_cached_data=True
        )

        val_dataset = CervicalSpineDataset(
            metadata_path=Config.VAL_METADATA, mode="val", load_cached_data=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=False,
        )

        return train_loader, val_loader

    def train_one_epoch(self, model, loader, optimizer, scheduler, scaler, epoch):
        """
        Runs one epoch of training with gradient accumulation and mixed precision.
        """
        model.train()
        running_loss = 0.0
        dataset_size = 0

        # Define Loss Function
        # Model outputs logits, so we use BCEWithLogitsLoss for AMP stability.
        # As per idea description, we do NOT use positive class weighting to maintain calibration.
        criterion = nn.BCEWithLogitsLoss()

        optimizer.zero_grad()

        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            # Mixed Precision Forward Pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / Config.ACCUMULATION_STEPS

            # Backward Pass
            scaler.scale(loss).backward()

            if (step + 1) % Config.ACCUMULATION_STEPS == 0:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                # Optimizer Step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Scheduler Step (Cosine Annealing usually updates per epoch,
                # but if OneCycle is used it updates per step.
                # Config implies CosineAnnealingLR with T_MAX, usually per epoch.
                # We will update scheduler in the epoch loop.)

            running_loss += (loss.item() * Config.ACCUMULATION_STEPS) * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, model, loader):
        """
        Evaluates the model on the validation set and computes the weighted log loss.
        """
        model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Mixed Precision Inference
                with autocast():
                    logits = model(images)
                    outputs = torch.sigmoid(logits)

                preds_list.append(outputs.cpu().numpy())
                targets_list.append(labels.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        # Calculate metric
        val_loss = weighted_log_loss(targets, preds)
        return val_loss

    def fit(self):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        # Initialize Data
        train_loader, val_loader = self.get_dataloaders()

        # Initialize Model
        model = CervicalFractureNet()
        model.to(self.device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Scaler for AMP
        scaler = GradScaler()

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, epoch
            )

            # Validate
            val_loss = self.validate(model, val_loader)

            # Step Scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss} | "  # Printing full precision as requested
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

            # Early Stopping and Model Saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")


def run_training():
    trainer = Trainer()
    trainer.fit()
