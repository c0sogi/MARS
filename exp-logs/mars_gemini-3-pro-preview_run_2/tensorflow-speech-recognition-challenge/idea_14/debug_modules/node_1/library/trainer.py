import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.network import EfficientNetV2Audio
from library.data_utils import load_all_data, load_background_noise, set_seed


class Trainer:
    """
    Trainer class for the Speech Command Recognition task.
    Manages GPU-resident data, training loops, evaluation, and submission generation.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Data containers (GPU resident)
        self.train_waveforms = None
        self.train_labels = None
        self.val_waveforms = None
        self.val_labels = None
        self.background_noise = None

        # Training state
        self.best_acc = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = 10

    def load_data(self):
        """
        Loads all datasets and moves them to the GPU.
        """
        print("Loading datasets to GPU...")

        # 1. Load Background Noise
        noise_tensor = load_background_noise(load_cached_data=True)
        self.background_noise = noise_tensor.to(self.device)

        # 2. Load Train Data
        train_wavs, train_lbls = load_all_data("train", load_cached_data=True)

        # Debug Mode: Subset data
        if Config.DEBUG:
            print(
                f"DEBUG mode: Truncating training data to {Config.DEBUG_SUBSET_SIZE} samples."
            )
            train_wavs = train_wavs[: Config.DEBUG_SUBSET_SIZE]
            train_lbls = train_lbls[: Config.DEBUG_SUBSET_SIZE]

        self.train_waveforms = train_wavs.to(self.device)
        self.train_labels = train_lbls.to(self.device)

        # 3. Load Val Data
        val_wavs, val_lbls = load_all_data("val", load_cached_data=True)

        if Config.DEBUG:
            val_wavs = val_wavs[: Config.DEBUG_SUBSET_SIZE]
            val_lbls = val_lbls[: Config.DEBUG_SUBSET_SIZE]

        self.val_waveforms = val_wavs.to(self.device)
        self.val_labels = val_lbls.to(self.device)

        print(
            f"Data loaded. Train: {self.train_waveforms.shape}, Val: {self.val_waveforms.shape}"
        )

    def setup_model(self):
        """
        Initializes model, optimizer, loss, and scheduler.
        """
        print("Initializing model...")
        self.model = EfficientNetV2Audio(
            background_noise=self.background_noise,
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def get_balanced_batch_indices(self):
        """
        Generates indices for a balanced epoch using weighted random sampling logic.
        Calculated on GPU.
        """
        # Calculate class counts
        labels = self.train_labels
        class_counts = torch.bincount(labels, minlength=Config.NUM_CLASSES).float()

        # Avoid division by zero
        class_counts = torch.where(
            class_counts > 0, class_counts, torch.ones_like(class_counts)
        )

        # Weights are inverse of frequency
        class_weights = 1.0 / class_counts

        # Assign weight to each sample
        sample_weights = class_weights[labels]

        # Sample indices
        num_samples = len(labels)
        indices = torch.multinomial(sample_weights, num_samples, replacement=True)

        return indices

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Get balanced indices
        indices = self.get_balanced_batch_indices()
        num_samples = indices.size(0)

        # Iterate over batches
        for i in range(0, num_samples, Config.BATCH_SIZE):
            batch_indices = indices[i : i + Config.BATCH_SIZE]

            x_batch = self.train_waveforms[batch_indices]
            y_batch = self.train_labels[batch_indices]

            self.optimizer.zero_grad()

            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """
        Runs validation on the held-out set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Process validation in batches
        num_samples = self.val_waveforms.size(0)
        indices = torch.arange(num_samples, device=self.device)

        with torch.no_grad():
            for i in range(0, num_samples, Config.BATCH_SIZE):
                batch_indices = indices[i : i + Config.BATCH_SIZE]

                x_batch = self.val_waveforms[batch_indices]
                y_batch = self.val_labels[batch_indices]

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * x_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return val_loss, val_acc

    def fit(self):
        """
        Main training loop with early stopping.
        """
        self.load_data()
        self.setup_model()

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Checkpointing
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  -> New best model saved! Accuracy: {val_acc}")
            else:
                self.epochs_no_improve += 1

            # Early Stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Starting inference on Test set...")

        # 1. Load Test Data
        # Using load_all_data will cache it as npy.
        test_wavs, _ = load_all_data("test", load_cached_data=True)

        # Load test metadata to get filenames
        df_test = pd.read_csv(Config.TEST_CSV)
        fnames = df_test["fname"].values

        # 2. Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("No best model found. Skipping inference.")
            return

        # Re-initialize model structure (noise injector can be empty for inference)
        model = EfficientNetV2Audio(
            background_noise=None, num_classes=Config.NUM_CLASSES, pretrained=False
        ).to(self.device)

        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        # Remove noise buffer to avoid shape mismatch between training (full noise) and inference (empty)
        if "noise_injector.noise_tensor" in state_dict:
            del state_dict["noise_injector.noise_tensor"]

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 3. Inference Loop
        all_preds = []
        num_samples = test_wavs.size(0)

        # Move test data to GPU
        print("Moving test data to GPU...")
        test_wavs = test_wavs.to(self.device)

        with torch.no_grad():
            for i in range(0, num_samples, Config.BATCH_SIZE):
                batch_wavs = test_wavs[i : i + Config.BATCH_SIZE]

                outputs = model(batch_wavs)
                # Get predicted class indices
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())

        # 4. Map to Labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

        # 5. Create Submission DataFrame
        df_sub = pd.DataFrame({"fname": fnames, "label": predicted_labels})

        # 6. Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(df_sub.head())
