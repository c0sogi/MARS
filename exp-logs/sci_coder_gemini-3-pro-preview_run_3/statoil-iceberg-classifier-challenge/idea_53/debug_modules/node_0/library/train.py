import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import process_data, get_dataloaders
from library.model import DIDPCNN

# Initialize Logger
logger = get_logger("train_module")


class Trainer:
    """
    Handles the training and validation loop for a single model instance.
    """

    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, loader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, angles, labels in loader:
            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, angles, labels in loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss


def run_training():
    """
    Executes the full 5-fold cross-validation training pipeline.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup directories
    Config.setup()

    logger.info("Starting training pipeline...")

    # Load and process data (cached)
    data = process_data(load_cached_data=True)

    # Store validation scores
    fold_scores = []

    # Cross-Validation Loop
    for fold in range(Config.NUM_FOLDS):
        logger.info(f"=== Starting Fold {fold} ===")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_dataloaders(
            data,
            fold_idx=fold,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = DIDPCNN().to(Config.DEVICE)

        # Optimizer (AdamW as per Idea)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Trainer
        trainer = Trainer(model, Config.DEVICE, criterion, optimizer)

        # Training Loop variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)

            logger.info(
                f"Fold {fold} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"New best model saved for Fold {fold} with Val Loss: {best_val_loss}"
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info(f"Early stopping triggered at Epoch {epoch + 1}")
                    break

        logger.info(f"Fold {fold} completed. Best Val Loss: {best_val_loss}")
        fold_scores.append(best_val_loss)

        # Clean up to save memory
        del model, optimizer, trainer, train_loader, val_loader
        torch.cuda.empty_cache()

    # Summary
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    logger.info("=== Cross-Validation Complete ===")
    logger.info(f"Fold Scores: {fold_scores}")
    logger.info(f"Average Log Loss: {avg_score} +/- {std_score}")
