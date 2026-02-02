import os
import torch
import torch.optim as optim
from library.config import Config
from library.utils import MaskedL1Loss
from library.model import train_epoch, validate, predict_and_submit


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    Encapsulates the training loop, optimization, scheduling, and early stopping.
    """

    def __init__(self, model, config=Config):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            config (class): Configuration class with hyperparameters.
        """
        self.model = model
        self.config = config
        self.device = config.DEVICE

        # Move model to the appropriate device (GPU/CPU)
        self.model.to(self.device)

        # Initialize Loss Function
        self.criterion = MaskedL1Loss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Initialize Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True,
        )

    def fit(self, train_loader, val_loader, epochs=None):
        """
        Execute the training loop with validation and early stopping.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int, optional): Number of epochs to train. Defaults to config.EPOCHS.
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            # Perform one epoch of training
            train_loss = train_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )

            # Perform validation
            val_loss = validate(self.model, val_loader, self.criterion, self.device)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Early Stopping and Model Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load the best model weights before finishing
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH))
            print("Loaded best model weights.")

    def predict(self, test_loader):
        """
        Generate predictions for the test set and save the submission file.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
        """
        # Delegate to the library function which handles inference and CSV saving
        predict_and_submit(self.model, test_loader, self.config)
