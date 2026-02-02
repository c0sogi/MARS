import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import TRAIN_CONFIG, PATHS, IDX_TO_LABEL
from library.dataset import SpeechCommandsDataset, get_balanced_dataframes
from library.model import ResNet34BiGRU
from library.utils import set_seed, calculate_accuracy


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the Speech Command model.
    """

    def __init__(self):
        """
        Initialize the Trainer.
        Sets up the device, data loaders, model, optimizer, loss function, and scheduler.
        """
        # 1. Setup Environment
        set_seed(TRAIN_CONFIG["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2. Prepare Data
        # Load dataframes using the cached/balanced loader from library.dataset
        self.df_train, self.df_val, self.df_test = get_balanced_dataframes(
            load_cached_data=True
        )

        # Handle Debug Mode
        if TRAIN_CONFIG["debug"]:
            print(
                f"Debug mode enabled. Sampling {TRAIN_CONFIG['debug_samples']} samples."
            )
            self.df_train = self.df_train.sample(
                n=min(len(self.df_train), TRAIN_CONFIG["debug_samples"]),
                random_state=TRAIN_CONFIG["seed"],
            ).reset_index(drop=True)
            self.df_val = self.df_val.sample(
                n=min(len(self.df_val), TRAIN_CONFIG["debug_samples"]),
                random_state=TRAIN_CONFIG["seed"],
            ).reset_index(drop=True)
            # We don't sample test set here to ensure predict() works on full set,
            # unless we specifically want to debug prediction too.

        # Create Datasets
        self.train_dataset = SpeechCommandsDataset(
            self.df_train, PATHS["train_audio_dir"], is_training=True
        )
        self.val_dataset = SpeechCommandsDataset(
            self.df_val, PATHS["train_audio_dir"], is_training=False
        )
        # Test dataset uses test_audio_dir
        self.test_dataset = SpeechCommandsDataset(
            self.df_test, PATHS["test_audio_dir"], is_training=False
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # 3. Initialize Model
        self.model = ResNet34BiGRU().to(self.device)

        # 4. Optimizer, Loss, Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TRAIN_CONFIG["num_epochs"]
        )

        # 5. Training State
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
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

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data).item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        return epoch_loss, epoch_acc

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets.data).item()
                total_samples += inputs.size(0)

        val_loss = running_loss / total_samples
        val_acc = running_corrects / total_samples

        return val_loss, val_acc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {TRAIN_CONFIG['num_epochs']} epochs...")

        for epoch in range(TRAIN_CONFIG["num_epochs"]):
            start_time = time.time()

            # Train and Validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), PATHS["model_save_path"])
                print(f"  -> New best model saved! (Val Acc: {val_acc})")
            else:
                self.patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {self.patience_counter}/{TRAIN_CONFIG['early_stopping_patience']}"
                )

            if self.patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_val_acc}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the submission file to disk.
        """
        print("Starting prediction on test set...")

        # Load best model weights
        if not os.path.exists(PATHS["model_save_path"]):
            print("Error: No model checkpoint found. Cannot predict.")
            return

        self.model.load_state_dict(
            torch.load(PATHS["model_save_path"], map_location=self.device)
        )
        self.model.eval()

        predictions = []
        fnames = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                predictions.extend(preds.cpu().numpy())

                # Get filenames for this batch
                # The dataset index aligns with the dataframe index
                # We need to calculate the indices corresponding to this batch
                start_idx = i * TRAIN_CONFIG["batch_size"]
                end_idx = start_idx + inputs.size(0)
                batch_fnames = (
                    self.df_test.iloc[start_idx:end_idx]["filepath"]
                    .apply(os.path.basename)
                    .tolist()
                )
                fnames.extend(batch_fnames)

        # Convert indices to labels
        predicted_labels = [IDX_TO_LABEL[idx] for idx in predictions]

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

        # Save to CSV
        submission_df.to_csv(PATHS["submission_path"], index=False)
        print(f"Submission saved to {PATHS['submission_path']}")
        print(f"Total predictions: {len(submission_df)}")
