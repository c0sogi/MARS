import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model_arch import HybridModel
from library.data_utils import get_dataloaders


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Manages the training, validation, and optimization loop.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                running_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs, early_stopping_patience):
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Scheduler step based on validation accuracy
            if self.scheduler:
                self.scheduler.step(val_acc)

            # Early Stopping Check
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model weights
        self.model.load_state_dict(best_model_wts)
        print(f"Best Validation Accuracy: {best_acc}")
        return self.model


def train_model():
    """
    Main function to setup and run the training pipeline.
    """
    set_seed(Config.SEED)

    # Load data using the cached data loader
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Initialize Model
    model = HybridModel(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        resnet_depth=Config.RESNET_DEPTH,
        resnet_width=Config.RESNET_WIDTH,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # ReduceLROnPlateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # Run Training
    model = trainer.fit(
        epochs=Config.EPOCHS, early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )

    return model, test_loader, test_ids


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    predictions = []
    device = Config.DEVICE

    print("Generating predictions...")
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            # Map 0-6 back to 1-7 (add 1)
            preds = predicted.cpu().numpy() + 1
            predictions.extend(preds)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
