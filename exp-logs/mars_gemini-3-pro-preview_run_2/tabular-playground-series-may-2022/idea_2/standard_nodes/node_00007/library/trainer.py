import torch
import torch.nn as nn
import torch.optim as optim
from library import config
from library.utils import get_device
from library.data_loader import get_dataloaders
from library.model import (
    WideMLP,
    train_model,
    generate_submission,
    train_one_epoch,
    validate,
)
from library.config import seed_everything


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Manufacturing Control task.
    """

    def __init__(
        self,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        seed=config.RANDOM_STATE,
    ):
        """
        Initializes the Trainer with model, optimizer, and criterion.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            seed (int): Random seed for reproducibility.
        """
        seed_everything(seed)
        self.device = get_device()

        # Initialize Model
        self.model = WideMLP().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Initialize Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        return train_one_epoch(
            self.model, loader, self.criterion, self.optimizer, self.device
        )

    def validate(self, loader):
        """
        Runs validation and returns loss and AUC.
        """
        return validate(self.model, loader, self.criterion, self.device)

    def fit(
        self,
        num_epochs=config.NUM_EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        load_cached_data=True,
    ):
        """
        Executes the full training pipeline with Early Stopping.

        Args:
            num_epochs (int): Maximum number of training epochs.
            patience (int): Early stopping patience.
            load_cached_data (bool): Whether to load pre-processed data from cache.

        Returns:
            DataLoader: The test data loader (for subsequent prediction).
        """
        # Load Data
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # Run Training Loop (handles early stopping and saving best model)
        self.model = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            num_epochs=num_epochs,
            patience=patience,
            save_path=config.MODEL_SAVE_PATH,
        )

        return test_loader

    def generate_submission(self, test_loader, output_path=config.SUBMISSION_SAVE_PATH):
        """
        Generates predictions for the test set and saves to CSV.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            output_path (str): Path to save the submission CSV.
        """
        generate_submission(self.model, test_loader, self.device, output_path)
