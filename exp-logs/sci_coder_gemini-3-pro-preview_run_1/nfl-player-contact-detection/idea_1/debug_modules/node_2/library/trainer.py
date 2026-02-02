import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.model import KinematicFFN
from library.dataset import create_dataloader
from library.data_processing import get_processed_dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # loss.item() is the mean loss of the batch. Multiply by batch size to get total.
        running_loss += loss.item() * features.size(0)
        count += features.size(0)

    return running_loss / count


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, true labels, and predicted probabilities.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * features.size(0)
            count += features.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / count
    # Concatenate and flatten to 1D arrays for metric computation
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    return avg_loss, all_targets, all_preds


def find_optimal_threshold(y_true, y_pred_probs):
    """
    Finds the probability threshold that maximizes MCC.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Search range from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        mcc = compute_mcc(y_true, y_pred_probs, threshold=thresh)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def generate_submission(model, scaler, threshold, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Load test data
    # Note: Test data usually doesn't have targets, so y will be None
    X_test, _, ids_test, _ = get_processed_dataset(
        mode="test",
        metadata_path=Config.TEST_METADATA_PATH,
        tracking_path=Config.TEST_TRACKING_PATH,
        scaler=scaler,
        load_cached_data=True,
    )

    # Create dataloader (no shuffle, large batch size for inference)
    test_loader = create_dataloader(
        X_test, y=None, batch_size=Config.BATCH_SIZE * 2, shuffle=False, pin_memory=True
    )

    model.eval()
    all_probs = []

    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_probs.append(outputs.cpu().numpy())

    all_probs = np.concatenate(all_probs).flatten()

    # Apply threshold to get binary predictions
    predictions = (all_probs >= threshold).astype(int)

    # Create DataFrame
    df_sub = pd.DataFrame({"contact_id": ids_test, "contact": predictions})

    # Save to file
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_pipeline(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Main training pipeline:
    1. Loads data
    2. Trains model with early stopping
    3. Optimizes threshold
    4. Generates submission
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading training data...")
    X_train, y_train, _, scaler = get_processed_dataset(
        mode="train",
        metadata_path=Config.TRAIN_METADATA_PATH,
        tracking_path=Config.TRAIN_TRACKING_PATH,
        load_cached_data=load_cached_data,
    )

    print("Loading validation data...")
    # Validation data uses the scaler fitted on training data
    # Note: Validation plays are in train_player_tracking.csv
    X_val, y_val, _, _ = get_processed_dataset(
        mode="val",
        metadata_path=Config.VAL_METADATA_PATH,
        tracking_path=Config.TRAIN_TRACKING_PATH,
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    # 2. Create DataLoaders
    train_loader = create_dataloader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model
    input_dim = X_train.shape[1]
    model = KinematicFFN(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        # Calculate MCC for monitoring (using default 0.5 threshold)
        val_mcc = compute_mcc(val_targets, val_probs, threshold=0.5)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCC (0.5): {val_mcc}"
        )

        # Early Stopping based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 5. Load Best Model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state.")

    # 6. Find Optimal Threshold
    print("Finding optimal threshold on validation set...")
    # Re-evaluate to get probabilities from the best model
    _, val_targets, val_probs = evaluate(model, val_loader, criterion, device)
    best_thresh, best_mcc = find_optimal_threshold(val_targets, val_probs)
    print(f"Optimal Threshold: {best_thresh} | Best Validation MCC: {best_mcc}")

    # 7. Generate Submission
    generate_submission(model, scaler, best_thresh, device)

    return model, best_thresh, best_mcc
