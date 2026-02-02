import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import SpeechCommandsDataset
from library.model import SpectroCNN


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
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


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
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


def generate_submission(model, device, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load Test Data
    test_dataset = SpeechCommandsDataset(mode="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    predictions = []
    fnames = []

    # Iterate over test data
    # Note: dataset returns (spec, label_id), but label_id is placeholder.
    # We need the filenames to construct the submission.
    # Accessing underlying dataframe for filenames is efficient.

    # We'll iterate the loader for predictions
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Get filenames from the dataset dataframe
    # The dataset df has 'filepath'. We need the basename.
    # e.g., test/audio/clip_00000.wav -> clip_00000.wav
    filepaths = test_dataset.df["filepath"].values
    fnames = [os.path.basename(fp) for fp in filepaths]

    # Map IDs to Labels
    predicted_labels = [Config.ID2LABEL[p] for p in predictions]

    # Create DataFrame
    submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Initializing Datasets...")
    train_dataset = SpeechCommandsDataset(
        mode="train", load_cached_data=load_cached_data
    )
    val_dataset = SpeechCommandsDataset(mode="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 2. Initialize Model
    model = SpectroCNN(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure cache dir exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} "
            f"| Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f} "
            f"| Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Load Best Model (but do not generate submission yet)
    print("Loading best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
