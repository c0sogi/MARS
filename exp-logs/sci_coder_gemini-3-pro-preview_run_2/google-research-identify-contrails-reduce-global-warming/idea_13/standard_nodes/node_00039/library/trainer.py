import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import MacroContextUNet
from library.loss import DeepSupervisionLoss
from library.metrics import GlobalDiceMetric


class Trainer:
    """
    Manages the training, validation, and checkpointing process for the Contrail Segmentation model.
    """

    def __init__(self, config: Config):
        """
        Initialize the Trainer with configuration settings.

        Args:
            config (Config): Configuration object containing hyperparameters and paths.
        """
        self.config = config
        self.device = config.device

        # Ensure reproducibility
        seed_everything(config.seed)

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize DataLoaders
        self.train_loader = get_dataloader(config, mode="train")
        self.valid_loader = get_dataloader(config, mode="validation")

        # Initialize Model
        self.model = MacroContextUNet(config)
        self.model.to(self.device)

        # Initialize Loss Function
        self.criterion = DeepSupervisionLoss(config)
        self.criterion.to(self.device)

        # Initialize Metric
        self.metric = GlobalDiceMetric(threshold=config.threshold)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_lr
        )

        # Training State
        self.best_score = -float("inf")
        self.current_epoch = 0

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            batch_size = images.size(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Returns a list of tensors [main, aux1, aux2] due to deep supervision
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Update stats
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        self.metric.reset()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, masks in self.valid_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                batch_size = images.size(0)

                # Forward pass
                # In eval mode, model returns a single tensor (main head)
                outputs = self.model(images)

                # Compute loss (Standard BCE+Dice on main output)
                loss = self.criterion(outputs, masks)

                # Update Metric
                self.metric.update(outputs, masks)

                # Update stats
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_dice = self.metric.compute()

        return epoch_loss, epoch_dice

    def fit(self, patience=7):
        """
        Main training loop with Early Stopping.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        print(
            f"Starting training for {self.config.epochs} epochs on device: {self.device}"
        )
        print(f"Model: {self.config.idea_name}")

        early_stopping_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch()

            # Validate
            val_loss, val_dice = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Print Metrics (Full precision for Val Dice)
            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Dice: {val_dice}"
            )

            # Checkpoint & Early Stopping
            if val_dice > self.best_score:
                self.best_score = val_dice
                save_path = self.config.get_model_save_path("best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> New Best Score! Model saved to {save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"  >>> No improvement. EarlyStopping counter: {early_stopping_counter}/{patience}"
                )

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Dice: {self.best_score}")


def main():
    # Initialize Config
    # Debug mode can be toggled here or via Config default
    config = Config(debug=False)

    # Initialize Trainer
    trainer = Trainer(config)

    # Start Training
    trainer.fit(patience=10)


# Note: The __name__ == "__main__" block is omitted as per instructions.
# To run this, one would import the Trainer class or call main() from a separate script.
