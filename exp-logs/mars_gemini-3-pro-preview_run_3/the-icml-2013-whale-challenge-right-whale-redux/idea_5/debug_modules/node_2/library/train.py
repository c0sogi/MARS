import os
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, get_logger, calculate_auc, calculate_pos_weight
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.loss import WeightedBCELoss, MixupLoss

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for Right Whale Detection.
    Handles training loop, validation, checkpointing, and inference.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

        # Initialize Model
        self.model = WhaleModel(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # Initialize Loss Function
        # Calculate positive class weight to handle imbalance
        pos_weight = calculate_pos_weight(Config.TRAIN_CSV)
        self.criterion = WeightedBCELoss(pos_weight=pos_weight, device=self.device)
        self.mixup_criterion = MixupLoss(self.criterion)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Training State
        self.best_val_auc = 0.0
        self.patience = 5  # Early stopping patience
        self.counter = 0
        self.best_model_path = os.path.join(
            Config.MODEL_CHECKPOINT_DIR, "best_model.pth"
        )

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for i, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Apply Mixup if enabled
            if Config.MIXUP:
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
                index = torch.randperm(inputs.size(0)).to(self.device)

                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                target_a, target_b = targets, targets[index]

                outputs = self.model(mixed_inputs)
                loss = self.mixup_criterion(outputs, target_a, target_b, lam)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Runs one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets, _ in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Apply sigmoid to get probabilities and flatten
                probs = torch.sigmoid(outputs).view(-1)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        auc = calculate_auc(all_targets, all_preds)

        return avg_loss, auc

    def fit(self):
        """
        Main training loop with early stopping and checkpointing.
        """
        logger.info(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_auc = self.validate_epoch()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - start_time

            # Print metrics with full precision
            logger.info(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {epoch_time:.2f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Checkpointing based on AUC
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with AUC: {val_auc}")
            else:
                self.counter += 1

            # Early Stopping
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        logger.info(f"Training complete. Best Validation AUC: {self.best_val_auc}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to submission.csv.
        """
        logger.info("Starting prediction on test set...")

        # Load best model weights
        if not os.path.exists(self.best_model_path):
            logger.warning("Best model not found. Using current model weights.")
        else:
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded best model from {self.best_model_path}")

        self.model.eval()
        all_clips = []
        all_probs = []

        with torch.no_grad():
            for inputs, _, clips in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).view(-1)

                all_clips.extend(clips)
                all_probs.extend(probs.cpu().numpy())

        # Create submission DataFrame
        df_sub = pd.DataFrame({"clip": all_clips, "probability": all_probs})

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
