import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import copy
import library.config as config
from library.model import ParallelDCNResNet


class Trainer:
    """
    Manages the training lifecycle with Early Stopping.
    """

    def __init__(self, device, input_dim):
        self.device = device
        self.input_dim = input_dim

        # Initialize Model
        self.model = ParallelDCNResNet(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_resnet_blocks=config.NUM_RESNET_BLOCKS,
            num_dcn_layers=config.NUM_DCN_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            num_classes=config.NUM_CLASSES,
        ).to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
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

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, train_loader, val_loader, epochs=config.EPOCHS):
        print(f"Starting training for {epochs} epochs...")

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Print metrics with full precision as requested
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            self.scheduler.step(val_acc)

            # Early Stopping Checkpoint
            # Cite solution_lesson_node_00003: Use deepcopy to prevent reference mutation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  -> New Best Model (Acc: {best_val_acc:.6f}). Saving...")
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, config.MODEL_PATH)

        print(f"Training complete. Best Validation Accuracy: {best_val_acc}")

    def predict(self, test_loader):
        print("Loading best model for predictions...")
        self.model.load_state_dict(torch.load(config.MODEL_PATH))
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                # Map 0-6 back to 1-7 (Target classes are 1-based)
                predicted = predicted + 1
                predictions.extend(predicted.cpu().numpy())

        # Load test IDs from metadata
        df_test = pd.read_parquet(config.TEST_PATH)
        ids = df_test[config.ID_COL].values

        # Create submission DataFrame
        submission = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: predictions})

        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
