import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, set_performance_mode
from library.data import get_dataloaders
from library.model import ZeroInitDeepAsymmetricNet


class Trainer:
    """
    Manages the training, validation, and inference processes.
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.FACTOR,
            patience=config.PATIENCE,
            verbose=True,
        )

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        return running_loss / total, correct / total

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        return running_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs):
        best_acc = 0.0
        best_model_state = None
        no_improve_epochs = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Step scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping Logic
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Use 2x patience for hard stopping to allow scheduler to work
            if no_improve_epochs >= self.config.PATIENCE * 2:
                print("Early stopping triggered.")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with Val Acc: {best_acc}")

        return best_acc

    def predict(self, loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                # Map 0-6 back to 1-7 (original class labels)
                predicted = predicted + 1
                predictions.extend(predicted.cpu().numpy())
        return predictions


def train_pipeline(
    debug_size=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Orchestrates the training pipeline: setup, data loading, training, and submission.
    """
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    # Disable strict determinism for performance (A100 optimization)
    set_performance_mode(deterministic=False, benchmark=True)
    device = Config.DEVICE

    # 2. Data Loading
    # get_dataloaders handles caching and processing internally
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, debug_size=debug_size
    )

    # Determine input dimension from data
    dummy_x, _ = next(iter(train_loader))
    input_dim = dummy_x.shape[1]

    # 3. Model Initialization
    model = ZeroInitDeepAsymmetricNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.RESNET_BLOCKS,
        dcn_layers=Config.DCN_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Training
    trainer = Trainer(model, device, Config)
    trainer.fit(train_loader, val_loader, epochs)

    # 5. Inference
    preds = trainer.predict(test_loader)

    # 6. Submission Generation
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
