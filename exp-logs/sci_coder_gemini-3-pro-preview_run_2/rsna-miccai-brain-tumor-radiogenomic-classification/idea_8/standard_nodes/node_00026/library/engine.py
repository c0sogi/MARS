import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library import config, utils, data, model


def set_seed(seed=config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate AUC, handling potential edge cases with single-class batches
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def predict_tta(model, dataloader, device):
    """
    Performs inference using Test-Time Augmentation (Original, H-Flip, V-Flip).
    Returns a dictionary mapping BraTS21ID to predicted probability.
    """
    model.eval()
    results = {}

    # Access IDs from the dataset dataframe to map predictions back to subjects
    ids = dataloader.dataset.df["BraTS21ID"].values
    current_idx = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # 1. Original Prediction
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is Width)
            inputs_h = torch.flip(inputs, [3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is Height)
            inputs_v = torch.flip(inputs, [2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average the probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            avg_probs = avg_probs.cpu().numpy().flatten()

            # Store results
            for i in range(batch_size):
                subject_id = ids[current_idx + i]
                results[subject_id] = float(avg_probs[i])

            current_idx += batch_size

    return results


def train_model(num_epochs=config.NUM_EPOCHS):
    """
    Main training loop with Early Stopping.
    """
    set_seed()
    config.setup_directories()
    device = torch.device(config.DEVICE)

    print(f"Device: {device}")
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Initialize Datasets
    # load_cached_anchors=True handles the caching requirement for ROI selection
    train_dataset = data.MGMTDataset(
        df_train, transforms=data.get_transforms("train"), load_cached_anchors=True
    )
    val_dataset = data.MGMTDataset(
        df_val, transforms=data.get_transforms("valid"), load_cached_anchors=True
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model, Loss, and Optimizer
    net = model.AsymmetricEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Early Stopping Variables
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Save Best Model based on Validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(net.state_dict(), config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs of no improvement."
            )
            break

    return best_auc


def generate_submission():
    """
    Generates predictions for the test set using the best model and saves to CSV.
    """
    set_seed()
    device = torch.device(config.DEVICE)

    print("Loading test metadata...")
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Initialize Test Dataset
    test_dataset = data.MGMTDataset(
        df_test, transforms=data.get_transforms("valid"), load_cached_anchors=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    net = model.AsymmetricEfficientNet().to(device)
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded best model from {config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model not found. Using initialized weights (random).")

    # Run Inference
    print("Running inference with TTA...")
    predictions = predict_tta(net, test_loader, device)

    # Format and Save Submission
    submission_df = pd.DataFrame(
        list(predictions.items()), columns=["BraTS21ID", "MGMT_value"]
    )
    submission_df = submission_df.sort_values("BraTS21ID")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
