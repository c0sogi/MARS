import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from collections import defaultdict

from library.config import Config
from library.utils import seed_everything, get_device, print_metric
from library.data import prepare_data
from library.model import WIISNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # Apply sigmoid for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC (handle edge case with single class in batch)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs one epoch of validation.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Main training pipeline:
    1. Prepares data
    2. Initializes model, optimizer, loss
    3. Runs training loop with Early Stopping
    4. Saves best model
    """
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_dataset, val_dataset, _ = prepare_data(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # Cite solution_lesson_node_00011: Implement worker_init_fn for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(Config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # 2. Initialize Model
    model = WIISNet()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print_metric("Train", "Loss", train_loss)
        print_metric("Train", "AUC", train_auc)
        print_metric("Val", "Loss", val_loss)
        print_metric("Val", "AUC", val_auc)

        # Early Stopping Logic based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved! AUC: {best_val_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")


def generate_submission():
    """
    Inference pipeline:
    1. Loads test data
    2. Loads best model
    3. Predicts on test slabs
    4. Aggregates predictions by Subject ID (Consensus Aggregation)
    5. Saves submission file
    """
    seed_everything(Config.SEED)
    device = get_device()

    # 1. Prepare Test Data
    _, _, test_dataset = prepare_data(load_cached_data=Config.LOAD_CACHED_DATA)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = WIISNet()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    print("Generating predictions on test set...")

    # 3. Prediction Loop
    subject_predictions = defaultdict(list)

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Map predictions to subject IDs
            # subject_ids is a tensor of shape (Batch,)
            current_ids = subject_ids.numpy()

            for sid, prob in zip(current_ids, probs):
                subject_predictions[sid].append(prob)

    # 4. Consensus Aggregation
    submission_data = []
    for sid, probs in subject_predictions.items():
        # Mean of predictions (1 value per subject now)
        mean_prob = np.mean(probs)
        submission_data.append({"BraTS21ID": sid, "MGMT_value": mean_prob})

    # 5. Save Submission
    df_submission = pd.DataFrame(submission_data)

    # Sort by ID for consistency
    df_submission = df_submission.sort_values("BraTS21ID")

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_submission.head())
