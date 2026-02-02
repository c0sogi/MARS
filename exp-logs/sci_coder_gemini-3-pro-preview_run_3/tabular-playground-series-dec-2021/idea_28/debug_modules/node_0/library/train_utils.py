import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model_utils import ParallelDCNResNet, set_seed
from library.data_utils import get_dataloaders


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    Saves the best model state using deepcopy.
    """

    def __init__(
        self,
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=False,
        delta=0,
        path=Config.MODEL_PATH,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path
        self.best_model_state = None

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy increases."""
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.val_acc_max} --> {val_acc}).  Saving model ..."
            )
        # Use deepcopy to prevent weight mutation issues
        self.best_model_state = copy.deepcopy(model.state_dict())
        torch.save(self.best_model_state, self.path)
        self.val_acc_max = val_acc


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
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
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def predict_and_submit(
    model, test_loader, test_ids, device, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions and saves the submission file.
    """
    print("Generating predictions...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Map back to 1-7 range (model predicts 0-6)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    print(f"Saving submission to {submission_path}...")
    df_sub.to_csv(submission_path, index=False)
    print("Submission saved.")


def run_training(
    epochs=Config.EPOCHS,
    max_train_samples=Config.MAX_TRAIN_SAMPLES,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
):
    """
    Main training driver. Initializes data, model, and runs the training loop.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Update Config for this run (allows debugging with smaller subsets)
    Config.MAX_TRAIN_SAMPLES = max_train_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # Initialize Model
    print(f"Initializing model with Input Dim: {input_dim}")
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE, verbose=True, path=Config.MODEL_PATH
    )

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print full precision as requested
        print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping step
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val Acc: {early_stopping.best_score}")

    # Load best model weights
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    # Generate and save submission
    predict_and_submit(model, test_loader, test_ids, device)

    return model
