import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import VentilatorDataset
from library.model import WideProjectedNet, train_epoch, validate


class Trainer:
    """
    Trainer class for the Wide-Projected Deeply-Supervised Physics-Identity Network.
    Orchestrates training, validation, and checkpointing.
    """

    def __init__(self, debug=None):
        """
        Initialize the Trainer.

        Args:
            debug (bool, optional): Overrides Config.debug if provided.
        """
        # Update Config if debug flag is passed
        if debug is not None:
            Config.debug = debug

        self.device = torch.device(Config.device)
        self._set_seed(Config.seed)

        # Initialize Data
        # VentilatorDataset handles feature engineering and caching internally
        self.train_ds = VentilatorDataset(split="train")
        self.val_ds = VentilatorDataset(split="val")

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        # Get input dimension from dataset (N, Seq, Features)
        # We access the tensor shape directly from the dataset
        input_dim = self.train_ds.x.shape[2]
        self.model = WideProjectedNet(input_dim=input_dim).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.lr_max, weight_decay=Config.weight_decay
        )

        # Initialize Scheduler (OneCycleLR)
        # Note: OneCycleLR requires the total number of steps to be known in advance
        steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.lr_max,
            epochs=Config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=Config.pct_start,
        )

    def _set_seed(self, seed):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def fit(self, epochs=None):
        """
        Execute the training loop.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to Config.epochs.
        """
        if epochs is None:
            epochs = Config.epochs

        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {Config.batch_size}")
        print(f"Gradient Clipping: {Config.clip_grad}")

        best_mae = float("inf")

        for epoch in range(epochs):
            # Train one epoch
            # train_epoch handles the forward pass, loss calculation (masked MAE + aux),
            # backward pass, gradient clipping, and optimizer/scheduler stepping.
            train_loss = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
            )

            # Validate
            # validate calculates the masked MAE on the validation set
            val_mae = validate(self.model, self.val_loader, self.device)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Checkpointing
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(self.model.state_dict(), Config.model_path)
                print(f"New best model saved! MAE: {best_mae}")

        print(f"Training complete. Best Validation MAE: {best_mae}")
