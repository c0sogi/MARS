import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SEED,
)
from library.utils import set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import DCHGNet
from library.loss import DCHGLoss


class Trainer:
    def __init__(self, model, criterion, optimizer, device, checkpoint_dir):
        """
        Args:
            model: The PyTorch model to train.
            criterion: The loss function module.
            optimizer: The optimizer.
            device: torch.device.
            checkpoint_dir: Directory to save model checkpoints.
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        # Metrics tracking
        metrics_sum = {}

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute loss
            loss, batch_metrics = self.criterion(outputs, targets, boundaries, mask)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for LSTMs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            running_loss += loss.item()

            # Accumulate metrics
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        avg_loss = running_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        metrics_sum = {}

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(features, mask)
                loss, batch_metrics = self.criterion(outputs, targets, boundaries, mask)

                running_loss += loss.item()

                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        avg_loss = running_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def fit(self, train_loader, val_loader, num_epochs, patience):
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device} for {num_epochs} epochs.")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(
                    f"Validation loss improved. Model saved to {self.best_model_path}"
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")
        return best_val_loss


def run_training(
    max_samples=None,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    load_cached_data=True,
    augment=True,
):
    """
    Main function to setup and run the training process.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached preprocessed data.
        augment (bool): Whether to apply data augmentation to training set.
    """
    set_seed(SEED)
    device = torch.device(DEVICE)

    # --- Data Loading ---
    print("Initializing datasets...")
    train_dataset = GestureDataset(
        split="train",
        load_cached_data=load_cached_data,
        max_samples=max_samples,
        augment=augment,
    )
    val_dataset = GestureDataset(
        split="val",
        load_cached_data=load_cached_data,
        max_samples=max_samples,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid potential multiprocessing issues in some envs
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- Model Setup ---
    print("Initializing model...")
    model = DCHGNet()

    # --- Loss & Optimizer ---
    criterion = DCHGLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # --- Execution ---
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=EARLY_STOPPING_PATIENCE,
    )
