import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TEST_CSV,
    SEED,
)
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    LabelEncoder,
)
from library.transforms import AudioProcessor
from library.modules import MultiResSKCRNN
from library.dataset import get_dataloaders


class Trainer:
    """
    Manages the training, evaluation, and prediction pipeline for the Multi-Resolution SK-CRNN model.
    """

    def __init__(self, learning_rate=LEARNING_RATE, device=None):
        set_seed(SEED)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model and Processor
        self.model = MultiResSKCRNN().to(self.device)
        self.processor = AudioProcessor().to(self.device)

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = None  # Initialized in fit()

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training with GPU-accelerated augmentation.
        """
        self.model.train()
        self.processor.train()  # Enables noise injection and SpecAugment

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # 1. GPU-Accelerated Augmentation & Feature Extraction
            # waveforms: (B, T), specs: (B, 3, F, T)
            waveforms_aug, specs_aug = self.processor(inputs)

            # 2. Forward Pass
            outputs = self.model(waveforms_aug, specs_aug)

            # 3. Loss & Backprop
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets)
                total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples

        return epoch_loss, epoch_acc

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set (no augmentation).
        """
        self.model.eval()
        self.processor.eval()  # Disables noise injection and SpecAugment

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Feature Extraction (No Augmentation)
                waveforms, specs = self.processor(inputs)

                # Forward Pass
                outputs = self.model(waveforms, specs)

                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets)
                total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples

        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=5,
        save_path=MODEL_SAVE_PATH,
    ):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        # Initialize scheduler based on total epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        best_val_acc = 0.0
        epochs_no_improve = 0

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(self.model, self.optimizer, epoch, val_acc, save_path)
                print(f"  -> New best model saved! (Acc: {best_val_acc})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  -> No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {best_val_acc}")
        return best_val_acc

    def generate_submission(
        self, test_loader, output_path=SUBMISSION_PATH, model_path=MODEL_SAVE_PATH
    ):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print(f"Generating submission using model at {model_path}...")

        # Load best model weights
        try:
            load_checkpoint(model_path, self.model, device=self.device)
        except FileNotFoundError:
            print(
                f"Error: Checkpoint file not found at {model_path}. Using current model state."
            )

        self.model.eval()
        self.processor.eval()

        all_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)

                # Process
                waveforms, specs = self.processor(inputs)
                outputs = self.model(waveforms, specs)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())

        # Decode labels
        label_encoder = LabelEncoder()
        pred_labels = label_encoder.decode_batch(all_preds)

        # Match with filenames from metadata
        df_test = pd.read_csv(TEST_CSV)
        fnames = df_test["filepath"].apply(os.path.basename).tolist()

        if len(fnames) != len(pred_labels):
            print(
                f"Warning: Mismatch between test files ({len(fnames)}) and predictions ({len(pred_labels)})."
            )

        # Save Submission
        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)

        print(f"Submission saved to {output_path}")
