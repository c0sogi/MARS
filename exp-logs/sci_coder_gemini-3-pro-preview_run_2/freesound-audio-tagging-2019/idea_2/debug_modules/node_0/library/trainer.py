import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_lwlrap
from library.dataset import mixup_data
from library.model import AudioClassifier


class Trainer:
    def __init__(self, train_loader, val_loader, test_loader=None):
        """
        Initializes the Trainer with dataloaders and model components.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = AudioClassifier(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model = self.model.to(self.device)

        # Loss Function (Multi-label classification)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Early Stopping & Checkpointing
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(Config.SAVE_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (specs, labels, _) in enumerate(self.train_loader):
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup if enabled
            if Config.USE_MIXUP:
                specs, labels_a, labels_b, lam = mixup_data(
                    specs,
                    labels,
                    alpha=Config.MIXUP_ALPHA,
                    use_cuda=(self.device.type == "cuda"),
                )
                outputs = self.model(specs)
                loss = lam * self.criterion(outputs, labels_a) + (
                    1 - lam
                ) * self.criterion(outputs, labels_b)
            else:
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), specs.size(0))

        return losses.avg

    def validate(self):
        """
        Runs validation on the validation set and computes LWLRAP.
        """
        self.model.eval()
        losses = AverageMeter()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for specs, labels, _ in self.val_loader:
                specs = specs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), specs.size(0))

                # Collect logits and targets for metric calculation
                # Move to CPU to avoid OOM on large validation sets
                all_targets.append(labels.cpu())
                all_outputs.append(outputs.cpu())

        # Concatenate all batches
        if len(all_targets) > 0:
            all_targets = torch.cat(all_targets, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)

            # Calculate LWLRAP
            # Note: calculate_lwlrap handles logits by sorting, which is monotonic with probability
            lwlrap = calculate_lwlrap(all_targets, all_outputs)
        else:
            lwlrap = 0.0

        return losses.avg, lwlrap

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.MAX_EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_lwlrap = self.validate()

            # Update Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics
            print(
                f"Epoch {epoch+1}/{Config.MAX_EPOCHS} "
                f"[Time: {elapsed:.2f}s] "
                f"Train Loss: {train_loss} "
                f"Val Loss: {val_loss} "
                f"Val LWLRAP: {val_lwlrap}"
            )

            # Early Stopping Logic
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def predict(self):
        """
        Generates predictions for the test set using the best model and saves submission.csv.
        """
        if self.test_loader is None:
            print("No test loader provided. Skipping prediction.")
            return

        print(f"Loading best model from {self.best_model_path} for inference...")
        if not os.path.exists(self.best_model_path):
            print(
                "Best model not found. Using current model state (warning: might not be best)."
            )
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()
        all_probs = []
        all_fnames = []

        print("Generating predictions...")
        with torch.no_grad():
            for specs, _, fnames in self.test_loader:
                specs = specs.to(self.device)

                # Forward pass
                logits = self.model(specs)
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_fnames.extend(fnames)

        if len(all_probs) == 0:
            print("No predictions generated.")
            return

        all_probs = np.concatenate(all_probs, axis=0)

        # Create submission DataFrame
        # Access classes from the dataset
        classes = self.test_loader.dataset.classes

        if not classes:
            print("Error: Classes not found in dataset.")
            return

        submission_df = pd.DataFrame(all_probs, columns=classes)
        submission_df.insert(0, "fname", all_fnames)

        # Save to CSV
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
