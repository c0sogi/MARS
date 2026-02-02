import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed, get_device, log_message, save_checkpoint
from library.data import get_dataloaders
from library.model import ModalityGroupedEfficientNet


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function: BCEWithLogitsLoss includes Sigmoid, stable for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: AdamW as requested
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: CosineAnnealingLR for stability
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Store predictions for AUC calculation
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.extend(targets.detach().cpu().numpy())
            all_probs.extend(probs)

        epoch_loss = running_loss / len(self.train_loader.dataset)

        # Step the scheduler
        if self.scheduler:
            self.scheduler.step()

        # Handle case where batch might contain only one class causing AUC error
        try:
            epoch_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            epoch_auc = 0.5

        duration = time.time() - start_time

        log_message(
            f"Epoch {epoch} [Train] - Loss: {epoch_loss}, AUC: {epoch_auc}, Time: {duration:.2f}s"
        )
        return epoch_loss, epoch_auc

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        start_time = time.time()

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs)

        epoch_loss = running_loss / len(self.val_loader.dataset)

        try:
            epoch_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            epoch_auc = 0.5

        duration = time.time() - start_time

        log_message(
            f"Epoch {epoch} [Val]   - Loss: {epoch_loss}, AUC: {epoch_auc}, Time: {duration:.2f}s"
        )
        return epoch_loss, epoch_auc

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        best_auc = 0.0
        patience_counter = 0

        log_message(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss, train_auc = self.train_epoch(epoch)
            val_loss, val_auc = self.validate(epoch)

            # Early Stopping Logic based on Validation AUC
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_auc, Config.MODEL_PATH
                )
                log_message(f"New best model found! AUC: {best_auc}")
            else:
                patience_counter += 1
                log_message(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break

        log_message(f"Training complete. Best Validation AUC: {best_auc}")


def train_model(debug=False):
    """
    Sets up the environment, data, and model, then executes training.

    Args:
        debug (bool): If True, runs on a subset of data for quick testing.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Loading
    log_message("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # 3. Model Initialization
    log_message("Initializing Modality-Aware 2.5D EfficientNet...")
    model = ModalityGroupedEfficientNet()
    model.to(device)

    # 4. Trainer Initialization
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Execute Training
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)
