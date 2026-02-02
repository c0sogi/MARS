import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import compute_auc
from library.model import mixup_data, mixup_criterion


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        """
        Initializes the Trainer with model, loaders, and optimization components.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize Loss Function with Class Weighting
        # Using explicit positive class weight to handle 1:9 imbalance
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

        # Training State
        self.best_auc = 0.0
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Early Stopping
        self.early_stopping_patience = 7
        self.patience_counter = 0

    def train_epoch(self):
        """
        Trains the model for one epoch using Mixup augmentation.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Apply Mixup
            data, target_a, target_b, lam = mixup_data(data, target, Config.MIXUP_ALPHA)

            self.optimizer.zero_grad()

            # Forward pass
            # Model output is (B, 1), squeeze to (B,)
            output = self.model(data).squeeze(1)

            # Compute Loss using Mixup Criterion
            loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def validate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data).squeeze(1)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                # Apply Sigmoid for probabilities
                probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / num_batches
        auc = compute_auc(all_targets, all_preds)

        return avg_loss, auc

    def fit(self, epochs=Config.EPOCHS):
        """
        Runs the full training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc} | "
                f"Time: {elapsed:.2f}s"
            )

            # Scheduler Step
            self.scheduler.step(val_auc)

            # Checkpointing and Early Stopping Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  >>> New Best AUC! Model saved to {self.best_model_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val AUC: {self.best_auc}")

    def load_best_model(self):
        """
        Loads the best model weights from the checkpoint.
        """
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model weights from {self.best_model_path}")
        else:
            print("Warning: Best model checkpoint not found. Using current weights.")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Returns:
            ids (list): List of clip IDs.
            probs (list): List of predicted probabilities.
        """
        self.model.eval()
        all_ids = []
        all_probs = []

        with torch.no_grad():
            for data, clip_ids in test_loader:
                data = data.to(self.device)

                output = self.model(data).squeeze(1)
                probs = torch.sigmoid(output).cpu().numpy()

                all_ids.extend(clip_ids)
                all_probs.extend(probs)

        return all_ids, all_probs

    def generate_submission(self, test_loader):
        """
        Generates predictions on the test set using the best model and saves to CSV.
        """
        # Ensure best model is loaded
        self.load_best_model()

        print("Generating predictions on Test set...")
        ids, probs = self.predict(test_loader)

        submission_df = pd.DataFrame({"clip": ids, "probability": probs})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
