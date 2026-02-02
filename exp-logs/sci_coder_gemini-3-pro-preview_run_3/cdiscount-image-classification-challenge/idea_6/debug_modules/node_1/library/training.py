import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    USE_MIXUP,
    MIXUP_ALPHA,
    ENSEMBLE_SIZE,
    SEED,
    TRAIN_FEATURES,
    TRAIN_LABELS,
    VAL_FEATURES,
    VAL_LABELS,
    CACHE_DIR,
    NUM_WORKERS,
)
from library.data_utils import FeatureDataset
from library.model import MLPClassifier


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, loader, criterion, optimizer, device, use_mixup, mixup_alpha
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, mixup_alpha, device
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # For accuracy calculation during training (approximate)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # We count correct if it matches the dominant label (lambda > 0.5 -> targets_a)
            if lam > 0.5:
                correct += predicted.eq(targets_a).sum().item()
            else:
                correct += predicted.eq(targets_b).sum().item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

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
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_ensemble(
    train_feat_path=TRAIN_FEATURES,
    train_label_path=TRAIN_LABELS,
    val_feat_path=VAL_FEATURES,
    val_label_path=VAL_LABELS,
    ensemble_size=ENSEMBLE_SIZE,
    output_dir=CACHE_DIR,
):
    """
    Trains an ensemble of MLP models on cached features.

    Args:
        train_feat_path (str): Path to training features .npy
        train_label_path (str): Path to training labels .npy
        val_feat_path (str): Path to validation features .npy
        val_label_path (str): Path to validation labels .npy
        ensemble_size (int): Number of models to train.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        list: A list of trained MLPClassifier models (loaded with best weights).
    """
    print(f"Loading features from {output_dir}...")

    if not os.path.exists(train_feat_path) or not os.path.exists(train_label_path):
        raise FileNotFoundError(
            "Training features/labels not found. Run feature extraction first."
        )

    # Load data into RAM (efficient for ~7M records * 1280 floats)
    X_train = np.load(train_feat_path)
    y_train = np.load(train_label_path)
    X_val = np.load(val_feat_path)
    y_val = np.load(val_label_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Create Datasets
    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)

    # Create DataLoaders
    # Using a large batch size for MLP training on features is usually efficient
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    trained_models = []
    os.makedirs(output_dir, exist_ok=True)

    for i in range(ensemble_size):
        model_seed = SEED + i
        print(
            f"\n=== Training Ensemble Model {i+1}/{ensemble_size} (Seed: {model_seed}) ==="
        )

        # Set seed for this model initialization
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        model = MLPClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_val_acc = -1.0
        patience_counter = 0
        best_model_path = os.path.join(output_dir, f"mlp_ensemble_{i}.pth")

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
                USE_MIXUP,
                MIXUP_ALPHA,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_acc)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc}"
            )

            # Early Stopping Check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc}"
                    )
                    break

        # Load best state
        print(f"Loading best weights for model {i+1} from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.eval()
        trained_models.append(model)

    return trained_models
