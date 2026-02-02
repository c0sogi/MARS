import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from library import config, utils, model, dataset


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction processes
    for the Multi-Resolution ResNet34-CRNN model.
    """

    def __init__(self):
        """
        Initialize the Trainer with device, model, loss function, optimizer, and scheduler.
        """
        # Ensure reproducibility
        utils.set_seed(config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = model.MultiResResNetCRNN(
            num_classes=config.NUM_CLASSES, pretrained=True
        )
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )

    def train_epoch(self, loader, max_batches=None):
        """
        Runs one epoch of training.

        Args:
            loader (DataLoader): The training data loader.
            max_batches (int, optional): Limit the number of batches for debugging.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for i, (inputs, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Handle case where loop was empty or broken early
        total_samples = len(all_targets)
        if total_samples == 0:
            return 0.0, 0.0

        epoch_loss = running_loss / total_samples
        epoch_acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, epoch_acc

    def validate(self, loader, max_batches=None):
        """
        Runs validation.

        Args:
            loader (DataLoader): The validation data loader.
            max_batches (int, optional): Limit the number of batches for debugging.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        total_samples = len(all_targets)
        if total_samples == 0:
            return 0.0, 0.0

        epoch_loss = running_loss / total_samples
        epoch_acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, epoch_acc

    def fit(self, num_epochs=config.NUM_EPOCHS, max_batches=None):
        """
        Main training loop with Early Stopping.

        Args:
            num_epochs (int): Number of epochs to train.
            max_batches (int, optional): Limit batches per epoch for debugging.
        """
        # Load Data
        print("Loading data...")
        train_loader, val_loader, test_loader = dataset.get_dataloaders(
            batch_size=config.BATCH_SIZE
        )

        best_acc = 0.0
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, max_batches=max_batches
            )
            val_loss, val_acc = self.validate(val_loader, max_batches=max_batches)

            # Update scheduler
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Early Stopping and Checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0

                # Save best model
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": best_acc,
                }
                utils.save_checkpoint(state, filename=config.MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {best_acc}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Generate submission after training
        self.predict(test_loader)

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves to CSV.

        Args:
            test_loader (DataLoader): The test data loader.
        """
        print("Generating submission...")

        # Load the best model weights
        checkpoint = utils.load_checkpoint(
            self.model, filename=config.MODEL_SAVE_PATH, device=self.device
        )

        if checkpoint is None:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        # Convert IDs to Labels
        predicted_labels = [config.ID2LABEL[p] for p in all_preds]

        # Get filenames from the dataset dataframe
        # Note: test_loader.dataset is a SpeechDataset, which has a .df attribute
        test_df = test_loader.dataset.df
        filenames = test_df["filepath"].apply(os.path.basename).tolist()

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"fname": filenames, "label": predicted_labels})

        # Save to CSV
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
