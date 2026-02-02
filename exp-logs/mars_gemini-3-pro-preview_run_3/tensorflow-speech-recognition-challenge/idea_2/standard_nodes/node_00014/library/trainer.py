import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.model import ResNetAudio


class Trainer:
    """
    Manages the training, validation, and inference of the audio classification model.
    """

    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (str): Compute device ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # CrossEntropyLoss in PyTorch (>=1.10) supports both:
        # 1. Target as indices (N,) -> Standard classification
        # 2. Target as probabilities (N, C) -> Soft targets (MixUp)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            # Note: targets are soft probabilities (Batch, NumClasses) due to MixUp
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Compute accuracy for monitoring (using argmax of soft targets)
            _, predicted = outputs.max(1)
            _, target_labels = targets.max(1)

            total += targets.size(0)
            correct += predicted.eq(target_labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """
        Runs evaluation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # Note: targets are class indices (Batch,) for validation
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device} for {num_epochs} epochs.")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, Config.MODEL_SAVE_PATH
                )
                print("New best model saved.")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        # Load best model weights
        try:
            load_checkpoint(Config.MODEL_SAVE_PATH, self.model, device=self.device)
            print("Loaded best model for inference.")
        except FileNotFoundError:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()

        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        predicted_ids = torch.argmax(all_probs, dim=1).numpy()

        # Map IDs to Labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in predicted_ids]

        # Get filenames from the dataset dataframe
        # We assume the loader preserves order (shuffle=False)
        test_df = test_loader.dataset.df
        fnames = test_df["filepath"].apply(os.path.basename).tolist()

        return fnames, predicted_labels


def run_training(train_loader, val_loader):
    """
    Helper function to initialize model and run training.
    """
    set_seed(Config.SEED)

    # Initialize Model
    model = ResNetAudio(num_classes=Config.NUM_CLASSES, pretrained=True)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Run Training
    trainer.fit()

    return trainer


def generate_submission(trainer, test_loader):
    """
    Generates the submission CSV file.
    """
    print("Generating submission...")
    fnames, labels = trainer.predict(test_loader)

    submission_df = pd.DataFrame({"fname": fnames, "label": labels})

    # Save to disk
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
