import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    set_seed,
    calculate_overall_lwlrap,
    mixup_data,
    mixup_criterion,
)
from library.model import ClassWiseEfficientNet
from library.dataset import get_dataloaders, get_label_map


class Trainer:
    """
    Manages the training, validation, and inference processes for the audio classification model.
    """

    def __init__(self, train_loader, val_loader, test_loader):
        self.device = Config.DEVICE
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize Model
        self.model = ClassWiseEfficientNet(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Loss Function
        # We use BCEWithLogitsLoss as this is a multi-label classification task
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (OneCycleLR)
        # Designed for super-convergence, cycling learning rate up and down
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.MAX_LR,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        # State tracking for Early Stopping
        self.best_model_state = None
        self.best_score = -np.inf

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Apply Mixup Augmentation
            # This regularizes the model by creating linear combinations of samples
            data, target_a, target_b, lam = mixup_data(
                data, target, alpha=Config.MIXUP_ALPHA, device=self.device
            )

            self.optimizer.zero_grad()
            output = self.model(data)

            # Compute Loss using Mixup Criterion
            loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluates the model on the validation set and computes LWLRAP.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target, _ in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()

                # Apply sigmoid to convert logits to probabilities
                preds = torch.sigmoid(output)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches for metric calculation
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Label-Weighted Label-Ranking Average Precision
        score = calculate_overall_lwlrap(all_targets, all_preds)

        return avg_loss, score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val LWLRAP: {val_score}"
            )

            # Early Stopping Logic
            # We monitor LWLRAP (higher is better)
            if val_score > self.best_score:
                self.best_score = val_score
                # Deepcopy to preserve the exact weights of the best epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save checkpoint
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.best_model_state, save_path)
                print(f"New best score! Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model weights for final inference
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with LWLRAP: {self.best_score}")

    def predict_test(self):
        """
        Generates predictions for the test set using the best trained model.
        """
        self.model.eval()
        all_preds = []
        all_fnames = []

        print("Generating predictions for test set...")
        with torch.no_grad():
            for data, fnames in self.test_loader:
                data = data.to(self.device)

                output = self.model(data)
                preds = torch.sigmoid(output)

                all_preds.append(preds.cpu().numpy())
                all_fnames.extend(fnames)

        all_preds = np.concatenate(all_preds, axis=0)
        return all_fnames, all_preds


def generate_submission(fnames, preds):
    """
    Formats the predictions into a CSV file matching the sample submission.
    """
    labels, label_map = get_label_map()

    # Create DataFrame
    # Ensure columns are in the correct order as per sample_submission.csv
    df = pd.DataFrame(preds, columns=labels)
    df.insert(0, "fname", fnames)

    # Ensure directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Orchestrates the data loading, training, and submission generation.
    """
    set_seed(Config.SEED)

    # Get DataLoaders (handles caching internally)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Execute Training
    trainer.fit()

    # Generate Predictions
    fnames, preds = trainer.predict_test()

    # Save Submission
    generate_submission(fnames, preds)
