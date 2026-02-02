import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda import amp

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, calculate_metrics
from library.dataset import get_dataloaders
from library.model import get_model


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction processes
    for the Herbarium 2020 plant species classification task.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = get_model(pretrained=Config.PRETRAINED)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Mixed Precision Scaler
        self.scaler = amp.GradScaler(enabled=(self.device == "cuda"))

        self.best_f1 = -1.0
        self.start_epoch = 0

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Automatic Mixed Precision Forward Pass
            with amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward Pass and Optimizer Step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with amp.autocast(enabled=(self.device == "cuda")):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / count

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate Macro F1
        f1 = calculate_metrics(all_labels, all_preds)

        return epoch_loss, f1

    def fit(self, load_cached_data=True):
        """
        Main training loop with early stopping and checkpointing.
        """
        set_seed(Config.SEED)

        # Load Data
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        print(f"Starting training on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        patience_counter = 0

        for epoch in range(self.start_epoch, Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_f1 = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.2f}s - LR: {current_lr}"
            )
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val F1: {val_f1}")

            # Checkpointing
            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                patience_counter = 0
                print(f"New best model found! Saving checkpoint.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_f1": self.best_f1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best=is_best,
            )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # After training, generate submission
        self.generate_submission(test_loader)

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        print("Loading best model for inference...")

        # Load best model weights
        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            checkpoint = torch.load(
                Config.MODEL_CHECKPOINT_PATH, map_location=self.device
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            print(f"Loaded model with Best F1: {checkpoint.get('best_f1', 'N/A')}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device, non_blocking=True)

                with amp.autocast(enabled=(self.device == "cuda")):
                    outputs = self.model(images)

                preds = torch.argmax(outputs, dim=1)

                all_ids.extend(image_ids.numpy())
                all_preds.extend(preds.cpu().numpy())

        # Create submission DataFrame
        submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

        # Sort by Id to match sample submission structure (optional but good practice)
        submission_df = submission_df.sort_values("Id").reset_index(drop=True)

        # Save to CSV
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
