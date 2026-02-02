import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
)
from library.data_loader import get_dataloaders
from library.model import CervicalFractureNet
from library.losses import WeightedMultiLabelLoss


class Trainer:
    """
    Manages the training, validation, and inference processes for the
    Cervical Spine Fracture Detection model.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the Trainer.

        Args:
            load_cached_data (bool): Whether to load dataset file paths from cache.
        """
        # 1. Setup Environment
        self.device = get_device()
        seed_everything(Config.SEED)

        # 2. Data Loaders
        # We retrieve all loaders. Test loader is stored for submission generation.
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # 3. Model
        self.model = CervicalFractureNet().to(self.device)

        # 4. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Loss Function
        # Config.POS_WEIGHT is used to weight the positive class (fracture) higher.
        self.criterion = WeightedMultiLabelLoss(
            pos_weight_value=Config.POS_WEIGHT, class_weights=Config.CLASS_WEIGHTS
        ).to(self.device)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        optimizer = self.optimizer
        optimizer.zero_grad()

        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(self.train_loader):
            # Move data to device
            images = data["image"].to(self.device)
            targets = data["target"].to(self.device)

            # Forward Pass
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            # Normalize loss for Gradient Accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS
            loss.backward()

            # Step Optimizer
            # We step if we reached accumulation steps OR if it's the last batch
            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0 or (
                batch_idx + 1
            ) == num_batches:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad()

            # Track Loss (scale back up to get the actual batch loss)
            running_loss += loss.item() * Config.GRAD_ACCUM_STEPS

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data in self.val_loader:
                images = data["image"].to(self.device)
                targets = data["target"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_val_loss = float("inf")
        patience = 3  # Number of epochs to wait for improvement
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(
            f"Epochs: {Config.EPOCHS}, Effective Batch Size: {Config.BATCH_SIZE * Config.GRAD_ACCUM_STEPS}"
        )

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train and Validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            duration = time.time() - start_time

            # Log Metrics
            metrics = {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Time": f"{duration:.2f}s",
            }
            print_metrics(metrics)

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, val_loss)
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at Epoch {epoch}.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model
        and saves them to submission.csv.
        """
        print("Generating submission...")

        # 1. Load Best Model
        checkpoint = load_checkpoint(
            self.model, filename=Config.MODEL_CHECKPOINT_PATH, device=self.device
        )
        if checkpoint is None:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        results = []
        # Target columns in the order output by the model
        target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        with torch.no_grad():
            for data in self.test_loader:
                images = data["image"].to(self.device)
                study_ids = data["study_id"]  # List of StudyInstanceUIDs

                # Forward Pass
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                # Map predictions to row_id format
                for i, study_uid in enumerate(study_ids):
                    # probs[i] is shape (8,)
                    for class_idx, class_name in enumerate(target_cols):
                        row_id = f"{study_uid}_{class_name}"
                        prob = float(probs[i][class_idx])

                        results.append({"row_id": row_id, "fractured": prob})

        # 2. Save to CSV
        df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
