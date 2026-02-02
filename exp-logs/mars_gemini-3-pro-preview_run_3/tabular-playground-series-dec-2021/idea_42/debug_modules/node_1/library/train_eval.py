import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import random
import sys

# Import from provided library files
from library.data_utils import load_and_preprocess_data
from library.model_arch import DualViewDCNResNet

# Set fixed random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Disable strict determinism for performance as per Lesson 00070
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Saves the best model state using deepcopy.
    """

    def __init__(self, patience=7, mode="max", min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        if mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, current_score, model):
        if self.mode == "min":
            score = -current_score
        else:
            score = current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model state when validation score improves."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_score = score

    def restore_best_weights(self, model):
        """Restores the best model weights."""
        if self.best_model_state is not None:
            print(f"Restoring best model weights with score: {self.val_score}")
            model.load_state_dict(self.best_model_state)
        else:
            print("No best model state to restore.")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(
    batch_size=4096,
    epochs=60,
    lr=1e-3,
    dcn_rank=4,
    dcn_layers=3,
    resnet_blocks=4,
    resnet_dim=512,
    dropout_rate=0.2,
    patience=10,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset, input_dim, num_classes, test_ids = (
        load_and_preprocess_data(load_cached_data=load_cached_data)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # Test loader will be used for inference later, but we can set it up here if needed

    # 2. Initialize Model
    print(
        f"Initializing DualViewDCNResNet (Input Dim: {input_dim}, Classes: {num_classes})..."
    )
    model = DualViewDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=dcn_rank,
        dcn_layers=dcn_layers,
        resnet_blocks=resnet_blocks,
        resnet_dim=resnet_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss()

    # Strictly use AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # ReduceLROnPlateau with aggressive decay factor of 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, verbose=True
    )

    early_stopping = EarlyStopping(patience=patience, mode="max")

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.8f}, Train Acc: {train_acc:.8f}, "
            f"Val Loss: {val_loss:.8f}, Val Acc: {val_acc:.8f}"
        )

        # Check early stopping
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Restore best weights
    early_stopping.restore_best_weights(model)

    return model, test_dataset, test_ids
