import os
import torch
import torch.nn as nn
from library.config import (
    DEVICE,
    MODELS_DIR,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
)
from library.utils import get_logger, seed_everything
from library.dataset import get_fold_dataloaders
from library.model import ExpertNet, train_one_epoch, validate


class Trainer:
    """
    Manages the training lifecycle for a single Expert model on a specific fold.
    Handles the training loop, validation, early stopping, and model saving.
    """

    def __init__(
        self,
        plane_name,
        fold_idx,
        model,
        optimizer,
        criterion,
        device,
        patience=5,
    ):
        self.plane_name = plane_name
        self.fold_idx = fold_idx
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience

        self.best_auc = -float("inf")
        self.counter = 0

        # Unique logger for this run
        self.logger = get_logger(f"train_{plane_name}_fold{fold_idx}")

        # Path to save the best model
        self.save_path = os.path.join(
            MODELS_DIR, f"best_model_{plane_name}_fold{fold_idx}.pth"
        )

    def fit(self, train_loader, val_loader, epochs):
        """
        Executes the training loop for the specified number of epochs.
        """
        self.logger.info(
            f"Starting training for {self.plane_name} - Fold {self.fold_idx}"
        )

        for epoch in range(1, epochs + 1):
            # Run training for one epoch
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )

            # Run validation
            val_loss, val_auc = validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Log metrics (Full precision for AUC as requested)
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping and Checkpointing Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                self.logger.info(f"New best model saved to {self.save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        self.logger.info(f"Training complete. Best Val AUC: {self.best_auc}")
        return self.best_auc


def run_training(
    plane_name,
    fold_idx,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=5,
):
    """
    Sets up and runs the training pipeline for a specific configuration.

    Args:
        plane_name (str): 'lower', 'center', or 'upper'.
        fold_idx (int): The fold index (0-4).
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for dataloaders.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.

    Returns:
        float: The best validation AUC achieved.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # 1. Prepare Data
    train_loader, val_loader = get_fold_dataloaders(
        fold_idx=fold_idx, plane_name=plane_name, batch_size=batch_size
    )

    # 2. Initialize Model
    # We use the defaults from config/model (EfficientNet-B0, pretrained)
    model = ExpertNet().to(DEVICE)

    # 3. Initialize Optimizer
    optimizer = model.get_optimizer(lr=lr, weight_decay=weight_decay)

    # 4. Initialize Loss Function
    # BCEWithLogitsLoss is standard for binary classification with raw logits
    criterion = nn.BCEWithLogitsLoss()

    # 5. Initialize Trainer
    trainer = Trainer(
        plane_name=plane_name,
        fold_idx=fold_idx,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        patience=patience,
    )

    # 6. Execute Training
    best_metric = trainer.fit(train_loader, val_loader, epochs)

    return best_metric
