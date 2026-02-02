import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_device
from library.model import ClassifierMLP, train_one_epoch, validate, generate_predictions
from library.features import FeaturePipeline
from library.dataset import ArenaDataset


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ModelTrainer:
    """
    Manages the training, validation, and prediction lifecycle of the model.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device, config
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=False,
            path=config.MODEL_SAVE_PATH,
        )

    def train_epoch(self):
        """Runs one epoch of training."""
        return train_one_epoch(
            self.model, self.train_loader, self.criterion, self.optimizer, self.device
        )

    def validate(self):
        """Runs validation."""
        return validate(self.model, self.val_loader, self.criterion, self.device)

    def train(self, epochs=None):
        """
        Executes the training loop with early stopping.
        Args:
            epochs (int, optional): Number of epochs to train. Defaults to Config.EPOCHS.
        """
        total_epochs = epochs if epochs is not None else self.config.EPOCHS
        print("Starting training...")

        for epoch in range(total_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Check early stopping
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Load best model weights
        print("Loading best model for inference...")
        self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH))

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        return generate_predictions(self.model, test_loader, self.device)


def run_training_task(
    debug_sample_size: int = None,
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    load_cached_data: bool = True,
):
    """
    Main function to execute the training pipeline.

    Args:
        debug_sample_size (int, optional): Number of samples to use for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        load_cached_data (bool): Whether to load features from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading & Processing
    pipeline = FeaturePipeline()
    X_train, y_train, X_val, y_val, X_test, test_ids = pipeline.process_data(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # Create Datasets
    train_dataset = ArenaDataset(X_train, y_train)
    val_dataset = ArenaDataset(X_val, y_val)
    test_dataset = ArenaDataset(X_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 3. Model Initialization
    model = ClassifierMLP(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    trainer = ModelTrainer(
        model, train_loader, val_loader, criterion, optimizer, device, Config
    )
    trainer.train(epochs=epochs)

    # 5. Inference and Submission
    print("Generating predictions on test set...")
    predictions = trainer.predict(test_loader)

    # Format Submission
    submission_df = pd.DataFrame(
        predictions, columns=["winner_model_a", "winner_model_b", "winner_tie"]
    )
    submission_df.insert(0, "id", test_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
