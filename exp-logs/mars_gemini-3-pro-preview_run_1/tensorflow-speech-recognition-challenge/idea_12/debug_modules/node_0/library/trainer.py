import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config, set_seed
from library.utils import map_predictions_to_submission


class Trainer:
    """
    Manages the training, validation, and inference processes for the Energy-Gated model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, label_encoder):
        """
        Args:
            model (nn.Module): The EnergyGatedEfficientNet model.
            train_loader (DataLoader): Loader for training data.
            val_loader (DataLoader): Loader for validation data.
            test_loader (DataLoader): Loader for test data.
            label_encoder (FineGrainedLabelEncoder): Encoder for handling label mappings.
        """
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_encoder = label_encoder

        # Optimization Setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # State tracking
        self.best_acc = 0.0
        self.work_dir = Config.WORK_DIR
        os.makedirs(self.work_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training with Mixup regularization.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (inputs, energy, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            energy = energy.to(self.device)
            targets = targets.to(self.device)

            # Mixup Regularization
            if Config.MIXUP_ALPHA > 0:
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
                index = torch.randperm(inputs.size(0)).to(self.device)

                # Mix inputs and energy vectors
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                mixed_energy = lam * energy + (1 - lam) * energy[index]
                target_a, target_b = targets, targets[index]

                # Forward pass
                outputs = self.model(mixed_inputs, mixed_energy)

                # Mixup Loss
                loss = lam * self.criterion(outputs, target_a) + (
                    1 - lam
                ) * self.criterion(outputs, target_b)
            else:
                outputs = self.model(inputs, energy)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        # Update Scheduler
        self.scheduler.step()

        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_loss:.6f}")

    def validate(self):
        """
        Evaluates the model on the validation set.
        Computes both Fine-Grained Accuracy and the Mapped Competition Accuracy.
        """
        self.model.eval()
        correct_fine = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, energy, targets in self.val_loader:
                inputs = inputs.to(self.device)
                energy = energy.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs, energy)
                _, predicted = torch.max(outputs.data, 1)

                # Fine-grained stats
                total += targets.size(0)
                correct_fine += (predicted == targets).sum().item()

                # Collect for mapping
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 1. Fine-Grained Accuracy (Internal Metric)
        fine_acc = correct_fine / total

        # 2. Mapped Accuracy (Competition Metric)
        # Convert IDs back to labels
        pred_labels = self.label_encoder.inverse_transform(all_preds)
        target_labels = self.label_encoder.inverse_transform(all_targets)

        # Map to 12-class target set (e.g., 'bed' -> 'unknown')
        mapped_preds = [self.label_encoder.map_to_target(l) for l in pred_labels]
        mapped_targets = [self.label_encoder.map_to_target(l) for l in target_labels]

        # Compute accuracy on mapped labels
        correct_mapped = sum(1 for p, t in zip(mapped_preds, mapped_targets) if p == t)
        mapped_acc = correct_mapped / len(mapped_targets)

        print(
            f"Validation - Fine-Grained Acc: {fine_acc:.10f}, Mapped Acc: {mapped_acc:.10f}"
        )
        return mapped_acc

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop. Saves the best model based on Mapped Accuracy.
        """
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.train_epoch(epoch)
            val_acc = self.validate()

            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint("best_model.pth")
                print(f"New best model saved with Mapped Acc: {val_acc:.10f}")

    def save_checkpoint(self, filename):
        """Saves model weights to the working directory."""
        path = os.path.join(self.work_dir, filename)
        torch.save(self.model.state_dict(), path)

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to submission.csv.
        """
        best_model_path = os.path.join(self.work_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: Best model not found. Using current weights.")

        self.model.eval()
        all_preds = []

        # Retrieve filenames from the dataset dataframe
        test_df = self.test_loader.dataset.df
        fnames = [os.path.basename(f) for f in test_df["filepath"]]

        with torch.no_grad():
            for inputs, energy, _ in self.test_loader:
                inputs = inputs.to(self.device)
                energy = energy.to(self.device)

                outputs = self.model(inputs, energy)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())

        # Map predictions to submission format (ids -> fine labels -> target labels)
        submission_labels = map_predictions_to_submission(all_preds, self.label_encoder)

        # Create submission DataFrame
        sub_df = pd.DataFrame({"fname": fnames, "label": submission_labels})

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
