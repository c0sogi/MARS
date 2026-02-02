import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import EfficientNetGeM


class Trainer:
    """
    Trainer class to handle training, validation, and inference for Right Whale Detection.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = EfficientNetGeM(pretrained=Config.PRETRAINED).to(self.device)

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # Loss Function with Class Weighting
        # Handle class imbalance by weighting the positive class
        # pos_weight must be a tensor on the device
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Tracking
        self.best_auc = 0.0

    def train_one_epoch(self):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            # Mixup Augmentation
            if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
                # Generate Mixup lambda from Beta distribution
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

                # Permute batch
                index = torch.randperm(inputs.size(0)).to(self.device)

                # Mix inputs
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

                # Targets for both parts of the mix
                targets_a, targets_b = targets, targets[index]

                # Forward pass
                outputs = self.model(mixed_inputs)

                # Calculate mixed loss
                # Mixing weighted scalar losses of the input pair
                loss = lam * self.criterion(outputs, targets_a) + (
                    1 - lam
                ) * self.criterion(outputs, targets_b)
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        # Update Scheduler (Cosine Annealing updates per epoch)
        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Calculate ROC AUC
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # Handle edge cases (e.g., only one class in batch)
            auc = 0.5

        return avg_loss, auc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch()
            val_loss, val_auc = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved! AUC: {val_auc}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Starting inference...")

        # Load best model weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading weights from {Config.BEST_MODEL_PATH}")
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model weights not found. Using current model state.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for inputs, clip_names in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                for clip, prob in zip(clip_names, probs):
                    results.append({"clip": clip, "probability": prob})

        # Save submission
        df_sub = pd.DataFrame(results)
        # Ensure column order matches requirements
        df_sub = df_sub[["clip", "probability"]]
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Entry point to run the training pipeline.
    """
    seed_everything(Config.SEED)
    trainer = Trainer()
    trainer.fit()
    trainer.predict()
