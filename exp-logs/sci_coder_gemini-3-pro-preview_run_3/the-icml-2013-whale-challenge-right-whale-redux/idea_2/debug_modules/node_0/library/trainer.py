import os
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import WhaleDataset
from library.model import SEResNet
from library.utils import set_seed, mixup_data, mixed_criterion, calculate_roc_auc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class Trainer:
    """
    Trainer class to manage training, validation, and inference for the Right Whale Detection task.
    """

    def __init__(self, debug=False):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, uses a small subset of data for debugging purposes.
        """
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)
        self.debug = debug

        # Initialize Datasets
        print("Initializing Datasets...")
        self.train_dataset = WhaleDataset(split="train", load_cached_data=True)
        self.val_dataset = WhaleDataset(split="val", load_cached_data=True)

        # Handle Debug Mode
        if self.debug:
            print("Debug mode enabled: Using subset of data.")
            # Use a small subset for quick debugging
            train_indices = list(range(min(len(self.train_dataset), 100)))
            val_indices = list(range(min(len(self.val_dataset), 50)))
            self.train_dataset = Subset(self.train_dataset, train_indices)
            self.val_dataset = Subset(self.val_dataset, val_indices)

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        print("Initializing Model...")
        self.model = SEResNet().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        # Loss Function (Weighted BCE)
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training using Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Mixup Augmentation
            data, target_a, target_b, lam = mixup_data(
                data, target, Config.MIXUP_ALPHA, self.device
            )

            # Forward pass
            output = self.model(data)

            # Mixed Loss
            loss = mixed_criterion(
                self.criterion, output, target_a.view(-1, 1), target_b.view(-1, 1), lam
            )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Track metrics (using primary target for approximation)
            with torch.no_grad():
                probs = torch.sigmoid(output)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(target_a.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)

        try:
            epoch_auc = calculate_roc_auc(all_targets, all_preds)
        except:
            epoch_auc = 0.5

        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss} - Train AUC: {epoch_auc}")
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target.view(-1, 1))

                running_loss += loss.item()

                probs = torch.sigmoid(output)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_auc = calculate_roc_auc(all_targets, all_preds)

        print(f"Validation - Loss: {val_loss} - AUC: {val_auc}")
        return val_loss, val_auc

    def fit(self, epochs=None):
        """
        Main training loop with Early Stopping.

        Args:
            epochs (int): Number of epochs to train. Defaults to Config.EPOCHS.
        """
        if epochs is None:
            epochs = Config.EPOCHS

        best_auc = 0.0
        patience_counter = 0

        print(f"Starting Training for {epochs} epochs...")

        for epoch in range(epochs):
            start_time = time.time()

            self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate()

            self.scheduler.step()

            # Checkpointing & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New Best AUC! Model saved to {self.best_model_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            print(f"Time: {time.time() - start_time}s")
            print("-" * 30)

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        print("Starting Inference on Test Set...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")
        else:
            print("Warning: No best model found. Using current model state.")

        self.model.eval()

        # Test Dataset
        test_dataset = WhaleDataset(split="test", load_cached_data=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        results = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                # If debug, limit inference
                if self.debug and batch_idx >= 5:
                    break

                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output).cpu().numpy().flatten()

                # Map global indices to clip names
                start_idx = batch_idx * Config.BATCH_SIZE
                for i, prob in enumerate(probs):
                    global_idx = start_idx + i
                    if global_idx < len(test_dataset):
                        clip_name = test_dataset.get_clip_name(global_idx)
                        results.append({"clip": clip_name, "probability": prob})

        df_submission = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

        print(f"Saving submission to {Config.SUBMISSION_FILE}...")
        df_submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print("Done.")
