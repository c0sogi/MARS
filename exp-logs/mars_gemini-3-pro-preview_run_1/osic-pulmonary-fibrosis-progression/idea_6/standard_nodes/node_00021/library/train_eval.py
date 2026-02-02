import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.loss import LaplaceLogLikelihoodLoss
from library.model import TQSAN


class Trainer:
    """
    Encapsulates the training and validation logic for the TQ-SAN model.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else Config.device

        # Loss function
        self.criterion = LaplaceLogLikelihoodLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_max, eta_min=Config.eta_min
        )

        # Training State
        self.best_score = -float("inf")
        self.model.to(self.device)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.model_save_path), exist_ok=True)

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
            # Move data to device
            axial = batch["axial"].to(self.device)
            coronal = batch["coronal"].to(self.device)
            meta = batch["meta"].to(self.device)
            target = batch["target"].to(self.device)
            dt = batch["dt"].to(self.device)
            base_fvc = batch["base_fvc"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(axial, coronal, meta)

            # Compute loss
            loss = self.criterion(preds, target, dt, base_fvc)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (weighted by batch size)
            batch_size = target.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return running_loss / n_samples

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Returns the average metric score (Negative Log Likelihood).
        """
        self.model.eval()
        total_metric = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                axial = batch["axial"].to(self.device)
                coronal = batch["coronal"].to(self.device)
                meta = batch["meta"].to(self.device)
                target = batch["target"].to(self.device)
                dt = batch["dt"].to(self.device)
                base_fvc = batch["base_fvc"].to(self.device)

                preds = self.model(axial, coronal, meta)

                # Calculate metric (higher is better)
                score = self.criterion.metric(preds, target, dt, base_fvc)

                batch_size = target.size(0)
                total_metric += score.item() * batch_size
                n_samples += batch_size

        return total_metric / n_samples

    def fit(self, epochs=Config.epochs, patience=Config.patience):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train and Validate
            train_loss = self.train_one_epoch(epoch)
            val_score = self.evaluate()

            # Step Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics (Full precision for validation score as requested)
            print(
                f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Val Score: {val_score}"
            )

            # Checkpoint and Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.model_save_path)
                print(f"  -> New Best Model Saved! Score: {self.best_score}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score}")


def train_model(train_dataset, val_dataset):
    """
    Helper function to set up DataLoaders, initialize the model, and run the training process.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.

    Returns:
        Trainer: The trainer instance containing the trained model.
    """
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = TQSAN()

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Execute Training
    trainer.fit()

    return trainer
