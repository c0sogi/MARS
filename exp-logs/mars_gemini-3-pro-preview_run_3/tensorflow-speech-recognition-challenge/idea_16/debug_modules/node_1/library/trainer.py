import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, MetricMonitor
from library.dataset import get_dataloaders
from library.model import FreqAttnResNeStCRNN
from library.audio_transforms import GPUAudioProcessor


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Speech Command Recognition task.
    """

    def __init__(self):
        # Setup directories and seed
        Config.setup()
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")

        # Data Loaders
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=True
        )

        # Model and Processor
        print("Initializing Model and Processor...")
        self.processor = GPUAudioProcessor().to(self.device)
        self.model = FreqAttnResNeStCRNN().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        self.processor.train()  # Enables augmentation

        metric_monitor = MetricMonitor()

        for batch_idx, (waveforms, targets) in enumerate(self.train_loader):
            waveforms = waveforms.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # 1. Process Audio (Augmentation + Spectrogram Generation)
            # Output: (B, 3, 224, 224)
            features = self.processor(waveforms)

            # 2. Forward Pass
            logits = self.model(features)

            # 3. Compute Loss
            loss = self.criterion(logits, targets)

            # 4. Backward Pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).float().mean()
                metric_monitor.update("loss", loss.item())
                metric_monitor.update("acc", accuracy.item())

        print(f"Epoch {epoch_idx} Train | {metric_monitor}")
        return metric_monitor.get_avg("loss"), metric_monitor.get_avg("acc")

    def validate(self, epoch_idx):
        """
        Runs validation loop.
        """
        self.model.eval()
        self.processor.eval()  # Disables augmentation

        metric_monitor = MetricMonitor()

        with torch.no_grad():
            for waveforms, targets in self.val_loader:
                waveforms = waveforms.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 1. Process Audio (No Augmentation)
                features = self.processor(waveforms)

                # 2. Forward Pass
                logits = self.model(features)

                # 3. Compute Loss
                loss = self.criterion(logits, targets)

                # Metrics
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).float().mean()
                metric_monitor.update("loss", loss.item())
                metric_monitor.update("acc", accuracy.item())

        print(f"Epoch {epoch_idx} Val   | {metric_monitor}")
        return metric_monitor.get_avg("loss"), metric_monitor.get_avg("acc")

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_acc = 0.0
        patience_counter = 0

        print("\nStarting Training...")
        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Update Scheduler
            self.scheduler.step()

            # Checkpoint & Early Stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                print(f"New best validation accuracy: {best_acc}")
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Validation Accuracy: {best_acc}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        """
        print("\nStarting Inference on Test Set...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            print(
                "No best model found. Using current model state (warning: might be untrained)."
            )
        else:
            print(f"Loading model from {self.best_model_path}")
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.processor.eval()

        predictions = []
        filenames = []

        # We need to map indices back to filenames.
        # The test_loader iterates sequentially, and the test_df in dataset.py is loaded from metadata/test.csv.
        # We reload the metadata here to ensure alignment.
        test_df = pd.read_csv(Config.TEST_CSV)
        if Config.DEBUG:
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Extract filenames from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
        # The submission format requires just the filename.
        test_filenames = test_df["filepath"].apply(os.path.basename).tolist()

        batch_start_idx = 0

        with torch.no_grad():
            for waveforms, _ in self.test_loader:
                waveforms = waveforms.to(self.device, non_blocking=True)

                # Process
                features = self.processor(waveforms)

                # Predict
                logits = self.model(features)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Map to labels
                batch_size = len(preds)
                for i in range(batch_size):
                    label_id = preds[i]
                    label_str = Config.ID2LABEL[label_id]

                    # Get corresponding filename
                    fname = test_filenames[batch_start_idx + i]

                    predictions.append({"fname": fname, "label": label_str})

                batch_start_idx += batch_size

        # Create DataFrame
        submission_df = pd.DataFrame(predictions)

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
