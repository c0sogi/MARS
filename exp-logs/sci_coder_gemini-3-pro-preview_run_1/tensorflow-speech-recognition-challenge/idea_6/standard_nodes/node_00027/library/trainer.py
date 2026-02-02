import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import PathConfig, TrainConfig, ModelConfig, DataConfig
from library.dataset import get_dataloaders
from library.model import DilatedEfficientNet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        self.device = torch.device(TrainConfig.DEVICE)
        set_seed(TrainConfig.SEED)

        # Load Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # Initialize Model
        self.model = DilatedEfficientNet().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TrainConfig.LEARNING_RATE,
            weight_decay=TrainConfig.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TrainConfig.T_MAX, eta_min=TrainConfig.ETA_MIN
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Training State
        self.best_acc = 0.0
        self.patience_counter = 0

    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply Mixup
            inputs, targets_a, targets_b, lam = self.mixup_data(
                inputs, targets, TrainConfig.MIXUP_ALPHA
            )

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.mixup_criterion(
                self.criterion, outputs, targets_a, targets_b, lam
            )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        print(f"Starting training on {self.device}...")

        for epoch in range(TrainConfig.EPOCHS):
            start_time = time.time()

            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{TrainConfig.EPOCHS} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Checkpoint and Early Stopping
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), PathConfig.MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {self.best_acc}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= TrainConfig.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(PathConfig.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(PathConfig.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No best model found. Using current model state.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        # Map IDs to Labels
        pred_labels = [DataConfig.ID2LABEL[p] for p in predictions]

        # Get filenames from dataset
        # We access the dataframe directly to get the filepaths corresponding to the loader order
        test_df = self.test_loader.dataset.dataframe
        fnames = test_df["filepath"].apply(os.path.basename).tolist()

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save
        os.makedirs(os.path.dirname(PathConfig.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(PathConfig.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {PathConfig.SUBMISSION_PATH}")


def run_training():
    trainer = Trainer()
    trainer.fit()
    trainer.generate_submission()
