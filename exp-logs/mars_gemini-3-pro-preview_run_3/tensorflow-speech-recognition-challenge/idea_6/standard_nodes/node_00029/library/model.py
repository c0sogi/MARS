import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, get_device, calculate_accuracy, MetricMonitor
from library.dataset import get_dataloaders
from library.sk_resnet import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, targets)
        metric_monitor.update("Loss", loss.item(), inputs.size(0))
        metric_monitor.update("Accuracy", accuracy, inputs.size(0))

    return metric_monitor.get_avg("Loss"), metric_monitor.get_avg("Accuracy")


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy = calculate_accuracy(outputs, targets)

            metric_monitor.update("Loss", loss.item(), inputs.size(0))
            metric_monitor.update("Accuracy", accuracy, inputs.size(0))

    return metric_monitor.get_avg("Loss"), metric_monitor.get_avg("Accuracy")


def run_training(
    epochs=Config.EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE, max_samples=None
):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        DataLoader: The test loader (to be used for submission).
    """
    set_seed(Config.SEED)
    device = get_device()

    # Dynamically update Config for debugging if max_samples is provided
    if max_samples is not None:
        Config.MAX_TRAIN_SAMPLES = max_samples

    # Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model
    print("Initializing Model...")
    model = get_model().to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f}")
        print(f"Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.10f}")
    return test_loader


def generate_submission(test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = get_device()
    model = get_model().to(device)

    # Load best weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No model checkpoint found. Using random weights.")

    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Map IDs to Labels
    pred_labels = [Config.ID2LABEL[p] for p in predictions]

    # Create Submission DataFrame
    # We read test.csv to get filenames.
    # Note: test_loader is created with shuffle=False, so the order matches test.csv rows.
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_csv_path)

    # Apply subsampling if configured, to match the test_loader
    if Config.DEBUG or (Config.MAX_TRAIN_SAMPLES is not None):
        n = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 200
        df_test = df_test.head(n)

    # Extract filename from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
    df_test["fname"] = df_test["filepath"].apply(os.path.basename)
    df_test["label"] = pred_labels

    submission_df = df_test[["fname", "label"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    """
    Main entry point to run the experiment.
    """
    Config.setup()
    test_loader = run_training()
    generate_submission(test_loader)
