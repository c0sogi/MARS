import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import TrainConfig, PathConfig
from library.utils import save_checkpoint, load_checkpoint, mixup_data, mixup_criterion


class Trainer:
    """
    Trainer class for the Right Whale Detection task.
    Handles Teacher and Student training phases, validation, and submission generation.
    """

    def __init__(self, model, train_loader, val_loader, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: Cosine Annealing for smooth convergence
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_lr
        )

        # Loss Function with Inverse Class Frequency Weighting
        # This addresses the imbalance between whale calls (minority) and noise (majority).
        self.criterion = self._get_criterion()

        # State tracking
        self.best_auc = 0.0

    def _get_criterion(self):
        """
        Calculates class weights based on training metadata and returns the Loss function.
        """
        try:
            train_df = pd.read_csv(PathConfig.train_meta)
            label_counts = train_df["label"].value_counts()
            neg_count = label_counts.get(0, 0)
            pos_count = label_counts.get(1, 0)

            if pos_count > 0:
                pos_weight_val = neg_count / pos_count
            else:
                pos_weight_val = 1.0
        except Exception as e:
            print(f"Warning: Could not calculate pos_weight from metadata: {e}")
            pos_weight_val = 1.0

        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        # BCEWithLogitsLoss combines Sigmoid and BCE.
        # pos_weight allows trading off recall and precision by upweighting the positive class.
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Apply Mixup
            # This works for both hard labels (Teacher) and soft labels (Student)
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, alpha=self.config.mixup_alpha, device=self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            # Squeeze to match target shape (Batch,)
            outputs = outputs.squeeze(1)

            # Calculate Loss using Mixup Criterion
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self):
        """
        Evaluates the model on the validation set using ROC AUC.
        """
        self.model.eval()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                # targets are kept on CPU for metric calculation

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).squeeze(1)

                all_targets.extend(targets.numpy())
                all_preds.extend(probs.cpu().numpy())

        # Handle edge case where batch might only have one class
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5

        return auc

    def train(self, save_path):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on {self.device} for {self.config.epochs} epochs.")
        early_stop_counter = 0
        self.best_auc = 0.0  # Reset for new training run

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_auc = self.validate()

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Checkpoint Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                early_stop_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_auc": self.best_auc,
                    },
                    save_path,
                )
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Explicitly reload the best model weights
        print(
            f"Training finished. Loading best model from {save_path} (AUC: {self.best_auc})"
        )
        load_checkpoint(self.model, save_path, device=self.device)

    def predict(self, loader):
        """
        Generates probability predictions for a given loader.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).squeeze(1)
                all_preds.extend(probs.cpu().numpy())

        return np.array(all_preds)

    def generate_submission(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Get predictions
        probs = self.predict(test_loader)

        # Get clip names from metadata
        # We assume the test_loader iterates in the same order as the metadata file
        # because library.dataset loads it sequentially and shuffle=False for test.
        try:
            test_df = pd.read_csv(PathConfig.test_meta)
            clip_names = test_df["clip_name"].values

            if len(probs) != len(clip_names):
                print(
                    f"Error: Prediction count ({len(probs)}) matches clip count ({len(clip_names)}) mismatch."
                )
                # Truncate to safe length to allow saving partial results if necessary
                min_len = min(len(probs), len(clip_names))
                probs = probs[:min_len]
                clip_names = clip_names[:min_len]

            submission = pd.DataFrame({"clip": clip_names, "probability": probs})

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            submission.to_csv(output_path, index=False)
            print(f"Submission saved to {output_path}")

        except Exception as e:
            print(f"Failed to generate submission: {e}")
