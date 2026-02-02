import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import TrainConfig, ModelConfig
from library.dataset import get_balanced_dataloader, get_test_dataloader, IDX2LABEL
from library.model import SKResNetBiGRU


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training():
    # Ensure reproducibility
    set_seed(TrainConfig.seed)

    # Setup directories
    TrainConfig.setup_directories()

    device = torch.device(TrainConfig.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_balanced_dataloader(
        TrainConfig.train_metadata_path,
        TrainConfig.batch_size,
        is_training=True,
        subset_size=TrainConfig.debug_subset_size if TrainConfig.debug else None,
    )

    val_loader = get_balanced_dataloader(
        TrainConfig.val_metadata_path,
        TrainConfig.batch_size,
        is_training=False,
        subset_size=TrainConfig.debug_subset_size if TrainConfig.debug else None,
    )

    # Model Setup
    print("Initializing Model...")
    model = SKResNetBiGRU().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=TrainConfig.lr, weight_decay=TrainConfig.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TrainConfig.epochs, eta_min=TrainConfig.min_lr
    )

    # Training Loop
    best_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(TrainConfig.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{TrainConfig.epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Checkpointing and Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), TrainConfig.model_save_path)
            print(f"New best model saved with accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")


def predict_and_submit():
    set_seed(TrainConfig.seed)
    device = torch.device(TrainConfig.device if torch.cuda.is_available() else "cpu")

    # Load Model
    print("Loading best model for inference...")
    model = SKResNetBiGRU().to(device)
    if os.path.exists(TrainConfig.model_save_path):
        model.load_state_dict(
            torch.load(TrainConfig.model_save_path, map_location=device)
        )
    else:
        print(
            "Warning: No trained model found. Using random initialization (likely to fail)."
        )

    model.eval()

    # Test Loader
    test_loader = get_test_dataloader(
        TrainConfig.test_metadata_path, TrainConfig.batch_size
    )

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, filepaths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            preds_cpu = preds.cpu().numpy()

            for pred_idx, filepath in zip(preds_cpu, filepaths):
                # Extract filename from path (e.g., test/audio/clip_001.wav -> clip_001.wav)
                fname = os.path.basename(filepath)
                label = IDX2LABEL[pred_idx]
                predictions.append({"fname": fname, "label": label})

    # Create DataFrame
    df_sub = pd.DataFrame(predictions)

    # Save Submission
    os.makedirs(os.path.dirname(TrainConfig.submission_path), exist_ok=True)
    df_sub.to_csv(TrainConfig.submission_path, index=False)
    print(f"Submission saved to {TrainConfig.submission_path}")
    print(df_sub.head())


if __name__ == "__main__":
    # Note: The prompt specifies NOT to include an if __name__ == "__main__": block
    # for execution in the module file itself, but to implement the module.
    # However, to make this file runnable as an entry point if needed, or
    # to strictly follow "implement the engine.py module", I provide the functions.
    # The user can import run_training and predict_and_submit.
    pass
