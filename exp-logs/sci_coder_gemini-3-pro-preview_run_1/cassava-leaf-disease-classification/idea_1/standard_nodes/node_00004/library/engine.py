import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_class_weights
from library.model import CassavaClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run on (cpu or cuda).

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        loader: The test DataLoader.
        device: The device to run on.

    Returns:
        list: A list of predicted class indices.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    return all_preds


class EarlyStopping:
    """
    Monitors validation metric to stop training early if it stops improving.
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        # We assume score is accuracy (higher is better)
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0


def run(
    train_loader,
    val_loader,
    test_loader,
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Main driver function for the training and inference pipeline.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """
    # 1. Setup
    Config.setup_directories()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 2. Model
    # Initialize model
    model = CassavaClassifier(num_classes=Config.NUM_CLASSES).to(device)

    # 3. Loss & Optimizer
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(Config.TRAIN_METADATA, debug=Config.DEBUG)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler: Reduce LR if validation accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    # 4. Training Loop
    print("Starting training...")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Freezing Strategy: Freeze backbone on first epoch, then unfreeze
        if epoch == 0:
            print("Freezing backbone layers...")
            model.freeze_backbone()
        elif epoch == 1:
            print("Unfreezing backbone layers...")
            model.unfreeze_backbone()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss} Accuracy: {train_acc}")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss} Accuracy: {val_acc}")

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Check
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Save Best Model
    if early_stopping.best_model_state is not None:
        print(f"Saving best model to {Config.MODEL_PATH}")
        torch.save(early_stopping.best_model_state, Config.MODEL_PATH)
        # Reload best weights for inference
        model.load_state_dict(early_stopping.best_model_state)
        # Move model back to device as state_dict was on CPU
        model.to(device)
    else:
        # Fallback if training didn't trigger early stopping logic
        torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Inference
    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # 7. Submission
    # Retrieve image IDs from the test dataset
    test_df = test_loader.dataset.df

    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
