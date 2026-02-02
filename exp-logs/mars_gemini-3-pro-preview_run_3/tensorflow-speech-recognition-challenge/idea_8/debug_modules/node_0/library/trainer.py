import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.feature_extractor import FeatureExtractor
from library.dataset import SpeechCommandDataset, get_weighted_sampler
from library.model import SKResNetConformer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the SKResNetConformer model.
    """

    def __init__(self):
        # 1. Setup Environment
        Config.setup()
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        self.best_acc = 0.0

        # 2. Prepare Data
        self._prepare_data()

        # 3. Initialize Model
        print(f"Initializing model: {Config.BACKBONE} + Conformer")
        self.model = SKResNetConformer().to(self.device)

        # 4. Define Loss, Optimizer, Scheduler
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

    def _prepare_data(self):
        """
        Loads metadata, caches features, and sets up DataLoaders.
        """
        print("Loading metadata...")
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        df_val = pd.read_csv(Config.VAL_METADATA)

        # Debugging / Subsampling if configured
        if Config.MAX_TRAIN_SAMPLES:
            df_train = df_train.sample(
                n=min(len(df_train), Config.MAX_TRAIN_SAMPLES), random_state=Config.SEED
            ).reset_index(drop=True)
        if Config.MAX_VAL_SAMPLES:
            df_val = df_val.sample(
                n=min(len(df_val), Config.MAX_VAL_SAMPLES), random_state=Config.SEED
            ).reset_index(drop=True)

        # Cache Features (Deterministic Processing)
        print("Caching features (Train)...")
        FeatureExtractor.cache_features(df_train, load_cached_data=True)
        print("Caching features (Val)...")
        FeatureExtractor.cache_features(df_val, load_cached_data=True)

        # Initialize Datasets
        self.train_dataset = SpeechCommandDataset(df_train, augment=True)
        self.val_dataset = SpeechCommandDataset(df_val, augment=False)

        # Initialize Sampler for Class Balancing
        train_sampler = get_weighted_sampler(df_train)

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        duration = time.time() - start_time

        return epoch_loss, epoch_acc, duration

    def validate(self):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def fit(self):
        """Main training loop with Early Stopping."""
        print("Starting training...")
        patience = 5
        counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_acc, duration = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update Scheduler
            self.scheduler.step()

            # Print Full Precision Metrics
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {duration:.2f}s")
            print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

            # Checkpoint & Early Stopping
            if val_acc > self.best_acc:
                print(
                    f"Validation Accuracy improved from {self.best_acc} to {val_acc}. Saving model..."
                )
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")

    def predict(self):
        """
        Generates predictions for the test set using the best model
        and saves them to submission.csv.
        """
        print("\n=== Starting Inference ===")

        # 1. Load Test Metadata
        if not os.path.exists(Config.TEST_METADATA):
            print("Test metadata not found. Skipping inference.")
            return

        df_test = pd.read_csv(Config.TEST_METADATA)

        # 2. Cache Test Features
        print("Caching features (Test)...")
        FeatureExtractor.cache_features(df_test, load_cached_data=True)

        # 3. Setup Test Loader
        test_dataset = SpeechCommandDataset(df_test, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 4. Load Best Model
        if not os.path.exists(Config.MODEL_PATH):
            print("Model file not found. Cannot predict.")
            return

        print(f"Loading best model from {Config.MODEL_PATH}")
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        # 5. Inference Loop
        predictions = []
        fnames = []

        # Extract filenames from dataframe (order is preserved by shuffle=False)
        # Assuming filepath format: "test/audio/clip_xxxx.wav"
        all_fnames = df_test["filepath"].apply(os.path.basename).tolist()

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                # Map indices to labels
                batch_preds = [Config.IDX_TO_LABEL[idx.item()] for idx in preds]
                predictions.extend(batch_preds)

        # 6. Save Submission
        submission = pd.DataFrame({"fname": all_fnames, "label": predictions})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
