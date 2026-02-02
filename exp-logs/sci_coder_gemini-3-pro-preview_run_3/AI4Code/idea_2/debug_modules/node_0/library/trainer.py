import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed
from library.dataset import NotebookSequenceDataset
from library.model import DSAPR


class ModelTrainer:
    """
    Manages the training and validation lifecycle of the DSAPR model.
    """

    def __init__(self, debug_limit: int = None):
        """
        Initialize the trainer with data, model, and optimization components.

        Args:
            debug_limit (int, optional): Limit the number of notebooks for debugging purposes.
        """
        # 1. Setup Environment
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        self.save_path = Config.MODEL_SAVE_PATH

        # Ensure working directory exists (redundant if Config handles it, but good practice)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        print(f"Initializing ModelTrainer on device: {self.device}")

        # 2. Prepare Data
        # We rely on the Dataset class to handle caching logic internally
        self.train_dataset = NotebookSequenceDataset(
            split="train", load_cached_data=True, debug_limit=debug_limit
        )
        self.val_dataset = NotebookSequenceDataset(
            split="val", load_cached_data=True, debug_limit=debug_limit
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        # 3. Initialize Model
        self.model = DSAPR().to(self.device)

        # 4. Optimization
        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Linear Warmup + Cosine Decay
        # OneCycleLR provides a good approximation of this strategy
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(self.train_loader),
            epochs=Config.NUM_EPOCHS,
            pct_start=Config.WARMUP_RATIO,
            anneal_strategy="cos",
        )

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in self.train_loader:
            # Move batch to device
            query = batch["query"].to(self.device)
            context = batch["context"].to(self.device)
            mask = batch["mask"].to(self.device)
            label = batch["label"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(query, context, mask)

            # Compute loss
            loss = self.criterion(preds, label)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            running_loss += loss.item() * query.size(0)
            count += query.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                query = batch["query"].to(self.device)
                context = batch["context"].to(self.device)
                mask = batch["mask"].to(self.device)
                label = batch["label"].to(self.device)

                preds = self.model(query, context, mask)
                loss = self.criterion(preds, label)

                running_loss += loss.item() * query.size(0)
                count += query.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Print metrics (full precision)
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train MSE: {train_loss} | Val MSE: {val_loss}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Validation loss improved. Model saved to {self.save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MSE: {best_val_loss}")
