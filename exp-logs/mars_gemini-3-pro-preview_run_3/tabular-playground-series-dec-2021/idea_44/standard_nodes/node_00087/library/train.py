import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_accuracy, save_submission
from library.data_loader import get_dataloaders
from library.model import AsymmetricParallelNet


def train_model(model, train_loader, val_loader, device):
    """
    Manages the training loop with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()

    # Strategy: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Strategy: ReduceLROnPlateau with aggressive decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy for monitoring
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {epoch_loss} Acc: {epoch_acc}")
        print(f"Val Loss: {val_loss} Acc: {val_acc}")

        # --- Scheduler Step ---
        scheduler.step(val_acc)

        # --- Early Stopping Logic ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint immediately
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def inference(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Convert 0-based indices back to 1-based Cover_Type
    predictions = np.array(predictions) + 1
    return predictions


def run_training():
    """
    Main execution function.
    """
    # 1. Initialization
    Config.initialize()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Retrieving dataloaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders()

    # Determine input dimension from a single batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    print(f"Detected Input Dimension: {input_dim}")

    # 3. Model Setup
    print("Initializing AsymmetricParallelNet...")
    model = AsymmetricParallelNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 4. Training
    model = train_model(model, train_loader, val_loader, device)

    # 5. Inference & Submission
    preds = inference(model, test_loader, device)
    save_submission(preds, test_ids, Config.SUBMISSION_PATH)
