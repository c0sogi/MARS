import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    calculate_accuracy,
    print_metrics,
)
from library.model import ConvNeXtAudio
from library.dataset import get_dataloaders


class Trainer:
    """
    Handles the training, validation, and inference processes for the Speech Command Recognition model.
    """

    def __init__(
        self,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        num_epochs=Config.NUM_EPOCHS,
        device=Config.DEVICE,
    ):
        self.device = device
        self.num_epochs = num_epochs

        # Initialize Model
        self.model = ConvNeXtAudio(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        # Loss Function: Cross Entropy with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc = calculate_accuracy(outputs, targets)
            batch_size = inputs.size(0)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total_samples += batch_size

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples

        return avg_loss, avg_acc

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                acc = calculate_accuracy(outputs, targets)
                batch_size = inputs.size(0)

                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                total_samples += batch_size

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples

        return avg_loss, avg_acc

    def fit(self, train_loader, val_loader, patience=5):
        """
        Main training loop with Early Stopping.
        """
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Print Metrics
            print_metrics("Train", epoch, train_loss, train_acc)
            print_metrics("Val", epoch, val_loss, val_acc)

            # Checkpoint and Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_acc,
                    path=Config.BEST_MODEL_PATH,
                )
                print(f"New best model saved with accuracy: {best_val_acc}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Loading best model for inference...")
        # Load best model weights
        checkpoint_data = load_checkpoint(
            self.model, path=Config.BEST_MODEL_PATH, device=self.device
        )
        if checkpoint_data is None:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        predictions = []
        fnames = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs, _, filenames in test_loader:
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Get predicted class indices
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                predictions.extend(preds)
                fnames.extend(filenames)

        # Map indices back to labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in predictions]

        # Create DataFrame
        df_submission = pd.DataFrame({"fname": fnames, "label": predicted_labels})

        # Save to CSV
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def run_training(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=5,
):
    """
    Entry point to run the full training and submission pipeline.
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Get DataLoaders
    print("Initializing DataLoaders...")
    loaders = get_dataloaders(batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 3. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        learning_rate=learning_rate, num_epochs=epochs, device=Config.DEVICE
    )

    # 4. Run Training
    print("Starting Training...")
    trainer.fit(train_loader, val_loader, patience=patience)

    # 5. Generate Submission
    print("Starting Inference...")
    trainer.generate_submission(test_loader)
