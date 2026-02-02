import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from library.config import Config
from library.utils import set_seed, LabelMapper
from library.model import SpeechCommandModel
from library.dataset import get_dataloaders


class Trainer:
    """
    Trainer class for the Speech Command Recognition model.
    Handles training with Mixup, validation with 12-class mapping, and submission generation.
    """

    def __init__(self, load_cached_data=True):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # 1. Prepare Data
        # load_cached_data=True allows skipping regeneration of balanced parquet if it exists
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # 2. Initialize Model
        self.model = SpeechCommandModel().to(self.device)

        # 3. Setup Optimization
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 4. Setup Scheduler
        # Phase 1: Linear Warmup (MIN_LR -> MAX_LR) for WARMUP_EPOCHS
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=Config.MIN_LR / Config.LEARNING_RATE,
            end_factor=1.0,
            total_iters=Config.WARMUP_EPOCHS,
        )

        # Phase 2: Cosine Annealing (MAX_LR -> MIN_LR) for remaining epochs
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS - Config.WARMUP_EPOCHS,
            eta_min=Config.MIN_LR,
        )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[Config.WARMUP_EPOCHS],
        )

        self.mapper = LabelMapper()
        self.best_acc = 0.0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training with Mixup regularization and Gradient Clipping.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # --- Mixup Regularization ---
            # We apply mixup to every batch as per strategy (alpha=1.0 implies strong mixing)
            alpha = Config.MIXUP_ALPHA
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0

            index = torch.randperm(inputs.size(0)).to(self.device)

            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            y_a, y_b = targets, targets[index]

            # --- Forward Pass ---
            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)

            # Mixup Loss
            loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(
                outputs, y_b
            )

            # --- Backward Pass ---
            loss.backward()

            # --- Gradient Clipping ---
            # Essential for RNN stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            running_loss += loss.item()

        # Step the scheduler at the end of the epoch
        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Validates the model.
        Computes the accuracy based on the mapping to the final 12 submission classes.
        """
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                # Get predicted class index (0-30)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # --- Compute 12-Class Accuracy ---
        # Map both predictions and targets to the submission format (e.g., 'bed' -> 'unknown')
        correct_12 = 0
        total_12 = len(all_targets)

        for p_idx, t_idx in zip(all_preds, all_targets):
            pred_label = self.mapper.index_to_submission(p_idx)
            true_label = self.mapper.index_to_submission(t_idx)

            if pred_label == true_label:
                correct_12 += 1

        acc_12 = correct_12 / total_12
        avg_loss = running_loss / len(self.val_loader)

        return avg_loss, acc_12

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")
        print(f"Batch Size: {Config.BATCH_SIZE}, Mixup Alpha: {Config.MIXUP_ALPHA}")

        patience = 10
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            duration = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:02d}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc (12-class): {val_acc:.10f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpoint and Early Stopping
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"  >>> New Best Model Saved! Accuracy: {val_acc:.10f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc:.10f}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        Saves the results to submission.csv.
        """
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(Config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded weights from {Config.CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted_indices = torch.max(outputs.data, 1)

                # Map indices to submission labels
                for idx in predicted_indices.cpu().numpy():
                    label = self.mapper.index_to_submission(idx)
                    predictions.append(label)

        # Match predictions with filenames
        # We read the test CSV again to ensure we have the filenames in the exact order of the loader
        # (since shuffle=False for test_loader)
        df_test = pd.read_csv(Config.TEST_CSV)

        # Extract filename from filepath (e.g., test/audio/clip_001.wav -> clip_001.wav)
        fnames = df_test["filepath"].apply(os.path.basename).tolist()

        if len(fnames) != len(predictions):
            raise ValueError(
                f"Mismatch: {len(fnames)} files vs {len(predictions)} predictions."
            )

        submission_df = pd.DataFrame({"fname": fnames, "label": predictions})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
