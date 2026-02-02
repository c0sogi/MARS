import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from library.config import Config
from library.utils import set_seed, calculate_lrap, save_checkpoint, load_checkpoint
from library.dataset import mixup_data
from library.model import AudioClassifier


class Trainer:
    def __init__(self, train_loader, val_loader, test_loader):
        """
        Initialize the Trainer with data loaders and model components.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize Model
        self.model = AudioClassifier().to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Loss Function: Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

        # Tracking
        self.best_score = -np.inf
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch):
        """
        Run one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Apply Mixup if enabled
            if Config.MIXUP:
                data, y_a, y_b, lam = mixup_data(
                    data, target, alpha=Config.MIXUP_ALPHA, device=self.device
                )
                outputs = self.model(data)
                loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(
                    outputs, y_b
                )
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)
            count += data.size(0)

        return running_loss / count

    def validate(self):
        """
        Run validation and calculate LRAP score.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                running_loss += loss.item() * data.size(0)
                count += data.size(0)

                # Apply sigmoid to get probabilities for metric calculation
                probs = torch.sigmoid(outputs)

                all_targets.append(target.cpu().numpy())
                all_predictions.append(probs.cpu().numpy())

        # Concatenate all batches
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        avg_loss = running_loss / count
        lrap_score = calculate_lrap(all_targets, all_predictions)

        return avg_loss, lrap_score

    def train(self, patience=7):
        """
        Full training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        epochs_no_improve = 0

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_score = self.validate()

            # Update scheduler
            self.scheduler.step()

            print(f"Epoch {epoch}/{Config.EPOCHS}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val LRAP: {val_score}")

            # Checkpoint and Early Stopping
            if val_score > self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving checkpoint."
                )
                self.best_score = val_score
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_score,
                    self.best_model_path,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement. Counter: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation LRAP: {self.best_score}")

    def predict(self):
        """
        Generate predictions for the test set and save submission file.
        """
        print("Starting prediction on test set...")

        # Load best model weights
        if os.path.exists(self.best_model_path):
            epoch, score = load_checkpoint(
                self.best_model_path, self.model, device=self.device
            )
            print(f"Loaded best model from epoch {epoch} with score {score}")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model state."
            )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # Prepare Submission DataFrame
        # Load test csv to get filenames and class column order
        test_df = pd.read_csv(Config.TEST_CSV)

        # Apply Debug slicing if enabled to match the Dataset behavior
        if Config.DEBUG:
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Identify class columns (exclude metadata columns)
        # Note: 'labels' is in meta_cols list but not in test.csv, which is fine
        meta_cols = ["fname", "labels", "filepath"]
        class_cols = [c for c in test_df.columns if c not in meta_cols]

        if all_probs.shape[1] != len(class_cols):
            raise ValueError(
                f"Prediction shape {all_probs.shape} does not match number of classes {len(class_cols)}"
            )

        # Create result DataFrame
        submission_df = test_df[["fname"]].copy()

        # Add probability columns
        prob_df = pd.DataFrame(all_probs, columns=class_cols)
        submission_df = pd.concat([submission_df, prob_df], axis=1)

        # Save to disk
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
