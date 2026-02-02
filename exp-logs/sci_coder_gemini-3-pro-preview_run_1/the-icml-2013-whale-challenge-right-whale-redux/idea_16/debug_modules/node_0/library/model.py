import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, calculate_auc
from library.dataset import get_dataloaders
from library.layers import ContextGatedResNet18

# Alias the model class to match the description if desired,
# though we use the imported class directly.
ContextGatedCRNN = ContextGatedResNet18


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # (B, 1) -> (B,)

        # Mixup Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            # Validation Loss (Standard BCE)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            # Store predictions for AUC
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

    avg_loss = running_loss / dataset_size
    auc = calculate_auc(all_targets, all_preds)

    return avg_loss, auc


def train_model_for_seed(seed, train_loader, val_loader, device, debug=False):
    """
    Trains a single model instance with a specific random seed.
    """
    print(f"--- Training with Seed {seed} ---")
    set_seed(seed)

    # Initialize Model
    model = ContextGatedCRNN(config=Config).to(device)

    # Loss Function with Positive Weighting
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
        verbose=False,
    )

    # Training Loop Variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.pth")

    epochs = Config.EPOCHS if not debug else 2

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_auc)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Val AUC: {val_auc:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE + 4:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best AUC for Seed {seed}: {best_auc:.10f}")
    return best_auc


def predict(model, loader, device):
    """
    Generates predictions for a given loader.
    """
    model.eval()
    predictions = []
    clips = []

    with torch.no_grad():
        for inputs, _, batch_clips in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

            predictions.extend(probs)
            clips.extend(batch_clips)

    return clips, predictions


def run_training_and_submission(load_cached_data=True, debug=False):
    """
    Main execution function:
    1. Loads data.
    2. Trains an ensemble of models (one per seed).
    3. Generates predictions on the test set.
    4. Saves the submission file.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    seeds = Config.SEEDS if not debug else [42]
    ensemble_probs = np.zeros(len(test_loader.dataset))
    test_clips = None

    # 2. Train Ensemble
    for seed in seeds:
        # Train
        train_model_for_seed(seed, train_loader, val_loader, device, debug)

        # Load Best Model for Inference
        model = ContextGatedCRNN(config=Config).to(device)
        model_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.pth")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(
                f"Warning: Model file for seed {seed} not found. Skipping inference for this seed."
            )
            continue

        # Inference
        print(f"Generating predictions for Seed {seed}...")
        clips, probs = predict(model, test_loader, device)

        if test_clips is None:
            test_clips = clips

        ensemble_probs += np.array(probs)

    # 3. Average Predictions
    avg_probs = ensemble_probs / len(seeds)

    # 4. Save Submission
    df_sub = pd.DataFrame({"clip": test_clips, "probability": avg_probs})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
