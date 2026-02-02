import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np

from library.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BEST_MODEL_PATH,
    NUM_WORKERS,
    METADATA_DIR,
    IDX_TO_LABEL,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import (
    set_seed,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    get_device,
)
from library.dataset import HybridAudioDataset
from library.transforms import SpecAugment, RawAudioAugment
from library.model import HybridDualStreamCRNN


class Trainer:
    """
    Encapsulates the training and validation logic for the Hybrid CRNN.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        early_stopping=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = early_stopping

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch_idx, (spec, wave, labels) in enumerate(train_loader):
            spec = spec.to(self.device)
            wave = wave.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(spec, wave)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        with torch.no_grad():
            for spec, wave, labels in val_loader:
                spec = spec.to(self.device)
                wave = wave.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(spec, wave)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs, 1)
                running_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        val_loss = running_loss / total_samples
        val_acc = running_correct / total_samples

        return val_loss, val_acc

    def fit(self, train_loader, val_loader, num_epochs):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = LEARNING_RATE

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"LR: {current_lr} - "
                f"Train Loss: {train_loss} - "
                f"Train Acc: {train_acc} - "
                f"Val Loss: {val_loss} - "
                f"Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if self.early_stopping:
                self.early_stopping(
                    val_loss, self.model, self.optimizer, self.scheduler, epoch
                )
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break


def run_training(
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Sets up datasets, model, and runs the training process.
    """
    set_seed(SEED)
    device = get_device()

    # 1. Prepare Datasets
    print("Initializing datasets...")

    # Augmentations for training
    spec_aug = SpecAugment()
    raw_aug = RawAudioAugment()

    train_dataset = HybridAudioDataset(
        metadata_file=os.path.join(METADATA_DIR, "train.csv"),
        spec_augment=spec_aug,
        raw_augment=raw_aug,
        is_test=False,
    )

    val_dataset = HybridAudioDataset(
        metadata_file=os.path.join(METADATA_DIR, "val.csv"),
        spec_augment=None,
        raw_augment=None,
        is_test=False,
    )

    # 2. Prepare Loaders with Weighted Sampling for Train
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Setup
    model = HybridDualStreamCRNN().to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, verbose=True, path=BEST_MODEL_PATH
    )

    # 4. Initialize Trainer and Fit
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        early_stopping=early_stopping,
    )

    trainer.fit(train_loader, val_loader, num_epochs=epochs)


def generate_submission(batch_size=BATCH_SIZE):
    """
    Loads the best model and generates predictions for the test set.
    """
    print("Generating submission...")
    set_seed(SEED)
    device = get_device()

    # 1. Load Data
    test_dataset = HybridAudioDataset(
        metadata_file=os.path.join(METADATA_DIR, "test.csv"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Load Model
    model = HybridDualStreamCRNN().to(device)
    try:
        load_checkpoint(BEST_MODEL_PATH, model, device=device)
        print(f"Loaded best model from {BEST_MODEL_PATH}")
    except FileNotFoundError:
        print("Best model not found. Ensure training has run.")
        return

    model.eval()

    predictions = []
    filenames = test_dataset.df["filepath"].apply(os.path.basename).tolist()

    # 3. Inference Loop
    with torch.no_grad():
        for spec, wave, _ in test_loader:
            spec = spec.to(device)
            wave = wave.to(device)

            outputs = model(spec, wave)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())

    # 4. Map to Labels
    predicted_labels = [IDX_TO_LABEL[p] for p in predictions]

    # 5. Create DataFrame and Save
    submission_df = pd.DataFrame({"fname": filenames, "label": predicted_labels})

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
