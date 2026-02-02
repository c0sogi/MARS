import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.dataset import get_dataloaders
from library.model import BreastCancerModel


class Trainer:
    """
    Manages the training and validation lifecycle of the Breast Cancer Detection model.
    Implements Analytical Prior Correction during validation.
    """

    def __init__(self):
        # 1. Setup Device and Seed
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")

        # 2. Data Loaders
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, _ = get_dataloaders()

        # 3. Model
        print("Initializing Model...")
        self.model = BreastCancerModel(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # 4. Optimization
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # 5. Analytical Correction Factor
        # Calculate the shift required to adjust logits from P_TRAIN (0.5) to P_TEST (~0.02)
        # Shift = - log(P_train/(1-P_train)) + log(P_test/(1-P_test))
        # Since P_train = 0.5, log(P_train/(1-P_train)) = log(1) = 0.
        p_train = Config.P_TRAIN
        p_test = Config.P_TEST

        term_train = np.log(p_train / (1 - p_train))
        term_test = np.log(p_test / (1 - p_test))

        self.logit_correction = term_test - term_train
        print(f"Analytical Logit Correction Factor: {self.logit_correction:.4f}")

        # 6. Checkpoints
        self.best_pf1 = -1.0
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass (Raw logits)
            logits = self.model(images)

            # Loss calculation
            # We train on raw logits against balanced labels to learn discrimination
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        duration = time.time() - start_time

        # Step the scheduler
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()

        print(
            f"Epoch [{epoch_idx+1}/{Config.EPOCHS}] Train Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | Time: {duration:.1f}s"
        )
        return avg_loss

    def validate(self):
        """
        Runs validation on the hold-out set.
        Applies Analytical Prior Correction to logits before calculating pF1.
        """
        self.model.eval()
        running_loss = 0.0

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass
                logits = self.model(images)

                # Calculate validation loss on raw logits (proxy for convergence)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                # Apply Analytical Correction for Metric Calculation
                # Corrects for the shift from Balanced Training (50%) to Natural Distribution (~2%)
                corrected_logits = logits + self.logit_correction
                probs = torch.sigmoid(corrected_logits)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_probs)

        # Calculate Probabilistic F1
        val_pf1 = probabilistic_f1(y_true, y_pred)

        print(f"Validation Results - Loss: {avg_loss:.6f} | pF1: {val_pf1:.10f}")
        return avg_loss, val_pf1

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"\nStarting training for {Config.EPOCHS} epochs...")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            _, val_pf1 = self.validate()

            # Checkpoint & Early Stopping
            if val_pf1 > self.best_pf1:
                print(
                    f"New Best pF1! ({self.best_pf1:.6f} -> {val_pf1:.6f}). Saving model..."
                )
                self.best_pf1 = val_pf1
                torch.save(self.model.state_dict(), self.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"\nTraining complete. Best Validation pF1: {self.best_pf1:.10f}")
        print(f"Best model saved to: {self.checkpoint_path}")


def run_training():
    """
    Entry point to run the training process.
    """
    trainer = Trainer()
    trainer.fit()
