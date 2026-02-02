import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import EarlyStopping
from library.model import train_one_epoch, validate


class Trainer:
    """
    Manages the training lifecycle of the Parallel DCN-ResNeXt model.
    Encapsulates optimizer setup, scheduling, and the training/validation loop
    with Early Stopping.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Args:
            model (nn.Module): The neural network to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE

        # Ensure model is on the correct device
        self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Initialize Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Initialize Early Stopping
        self.early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="max")

    def fit(self):
        """
        Executes the training loop for the configured number of epochs.

        Returns:
            nn.Module: The model with the best validation weights loaded.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # 1. Training Step
            train_loss, train_acc = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

            # 2. Validation Step
            val_loss, val_acc = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # 3. Scheduler Step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 4. Logging (Full Precision)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # 5. Early Stopping Check
            self.early_stopping(val_acc, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # 6. Load Best Weights
        print("Loading best model weights...")
        self.early_stopping.load_best_weights(self.model)

        return self.model
