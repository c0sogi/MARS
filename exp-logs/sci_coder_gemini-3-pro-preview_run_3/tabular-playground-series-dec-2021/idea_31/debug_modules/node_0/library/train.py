import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
import sys
from library.config import config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders
from library.model import ParallelDCNResNet


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the model.
    Implements AdamW, ReduceLROnPlateau, and custom Early Stopping.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # Scheduler: Reduce LR when validation accuracy plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience,
            verbose=False,  # We will manually print LR if needed or rely on implicit updates
        )

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Runs evaluation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

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
        """
        Main training loop with Early Stopping and Scheduler.
        """
        best_val_acc = -np.inf
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        early_stopping_patience = config.train.early_stopping_patience

        print(f"Starting training on device: {self.device}")

        for epoch in range(config.train.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update Scheduler based on Validation Accuracy
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_acc)
            new_lr = self.optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch+1}/{config.train.epochs}")
            print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")
            if new_lr != old_lr:
                print(f"Learning Rate changed from {old_lr} to {new_lr}")

            # Early Stopping Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # Save checkpoint immediately
                torch.save(best_model_wts, config.paths.model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Accuracy: {best_val_acc}")

        # Load best weights
        self.model.load_state_dict(best_model_wts)
        return best_val_acc

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())

        # Map 0-6 back to 1-7
        final_preds = np.array(all_preds) + 1
        return final_preds


def run_training(debug=False):
    """
    Orchestrates the entire training pipeline.
    """
    # 1. Reproducibility
    seed_everything(config.train.seed)

    # 2. Data Loading
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 3. Model Initialization
    device = torch.device(config.train.device)
    model = ParallelDCNResNet().to(device)

    # 4. Trainer Initialization
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Training
    trainer.fit()

    # 6. Inference
    print("Generating predictions for test set...")
    predictions = trainer.predict(test_loader)

    # 7. Submission
    save_submission(test_ids, predictions, config.paths.submission_path)


if __name__ == "__main__":
    # This block is included for local testing if run directly,
    # but the function run_training is the entry point.
    run_training()
