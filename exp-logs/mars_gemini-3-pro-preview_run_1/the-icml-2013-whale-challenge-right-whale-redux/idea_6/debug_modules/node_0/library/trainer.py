import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_auc, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, mixup_data, mixup_criterion
from library.model import TimePreservingEfficientNetBiGRU


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the whale detection model.
    """

    def __init__(self):
        self.device = Config.DEVICE

        # Initialize Model
        self.model = TimePreservingEfficientNetBiGRU().to(self.device)

        # Loss Function with Class Imbalance Handling
        # Explicitly handling the ~1:9 imbalance
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Apply Mixup
            data, target_a, target_b, lam = mixup_data(
                data, target, Config.MIXUP_ALPHA, self.device
            )

            self.optimizer.zero_grad()
            output = self.model(data)

            # Calculate Mixup Loss
            # BCEWithLogitsLoss expects targets to be same shape as output (Batch, 1)
            loss = mixup_criterion(
                self.criterion,
                output,
                target_a.unsqueeze(1),
                target_b.unsqueeze(1),
                lam,
            )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # Validation Loss (Standard BCE)
                loss = self.criterion(output, target.unsqueeze(1))
                running_loss += loss.item()

                # Store predictions for AUC
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                targets = target.cpu().numpy().flatten()

                all_preds.extend(probs)
                all_targets.extend(targets)

        val_loss = running_loss / len(val_loader)
        val_auc = calculate_auc(all_targets, all_preds)

        return val_loss, val_auc

    def train(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        best_auc = 0.0
        patience = 5
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Step Scheduler based on AUC
            self.scheduler.step(val_auc)

            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_auc, best_model_path
                )
                patience_counter = 0
                print(f"New best model saved! AUC: {best_auc:.6f}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # Load best model for final state
        print("Loading best model for inference...")
        load_checkpoint(best_model_path, self.model, device=self.device)

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        self.model.eval()
        results = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for data, clip_ids in test_loader:
                data = data.to(self.device)
                output = self.model(data)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(output).cpu().numpy().flatten()

                for clip_id, prob in zip(clip_ids, probs):
                    results.append({"clip": clip_id, "probability": prob})

        # Create Submission DataFrame
        df_submission = pd.DataFrame(results)

        # Save to file
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        return df_submission


def run_training():
    """
    Entry point to run the full training and prediction pipeline.
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Get DataLoaders (handles caching)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Train Model
    trainer.train(train_loader, val_loader)

    # 5. Generate Submission
    trainer.predict(test_loader)
