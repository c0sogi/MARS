import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data_processor, model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, best MCC, and best threshold.
    """
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(features)
            loss = criterion(logits, targets)

            running_loss += loss.item() * features.size(0)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Concatenate results
    y_logits = np.vstack(all_logits)
    y_true = np.vstack(all_targets)

    # Convert logits to probabilities
    y_probs = 1.0 / (1.0 + np.exp(-y_logits))

    # Optimize threshold for MCC
    best_thresh = utils.optimize_threshold(y_true, y_probs)

    # Calculate MCC at best threshold
    y_pred = (y_probs >= best_thresh).astype(int)
    mcc = utils.matthews_corrcoef(y_true, y_pred)

    return avg_loss, mcc, best_thresh


def train_model(debug_sample=config.DEBUG_SAMPLE_SIZE):
    """
    Main function to train the IP-RVN model.
    """
    utils.seed_everything()
    device = torch.device(config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, _, _ = data_processor.prepare_datasets(
        load_cached_data=True, debug_sample=debug_sample
    )

    # 2. Get Feature Names for Model Initialization
    # We read the columns from the cached parquet file to ensure we have the exact feature list
    # used by the data processor.
    train_parquet_path = os.path.join(config.WORKING_DIR, "train_features.parquet")
    if not os.path.exists(train_parquet_path):
        raise FileNotFoundError(
            f"Training cache not found at {train_parquet_path}. Run data prep first."
        )

    # Read just the columns (schema) to save memory
    df_schema = pd.read_parquet(train_parquet_path).head(0)
    feature_names = data_processor.get_feature_columns(df_schema.columns)

    print(f"Initializing IP-RVN with {len(feature_names)} features.")

    # 3. Initialize Model, Loss, Optimizer
    net = model.IPRVN(feature_names).to(device)
    criterion = utils.FocalLoss()
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_mcc = -1.0
    best_epoch = 0
    patience_counter = 0

    model_save_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    thresh_save_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")

    print("Starting training...")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_mcc, val_thresh = evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCC: {val_mcc} | "
            f"Thresh: {val_thresh}"
        )

        # Checkpoint
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_epoch = epoch
            patience_counter = 0

            # Save Model
            torch.save(net.state_dict(), model_save_path)
            # Save Threshold
            np.save(thresh_save_path, np.array([val_thresh]))
            print(f"  -> New best model saved! MCC: {best_mcc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best MCC: {best_mcc} at Epoch {best_epoch}")


def predict():
    """
    Loads the best model and generates predictions for the test set.
    """
    utils.seed_everything()
    device = torch.device(config.DEVICE)

    # 1. Load Data
    # We need the test loader and the test dataframe (for IDs)
    _, _, test_loader, df_test = data_processor.prepare_datasets(load_cached_data=True)

    if test_loader is None or df_test is None:
        print("No test data found. Skipping inference.")
        return

    # 2. Get Feature Names
    # Same logic as training to ensure architecture match
    train_parquet_path = os.path.join(config.WORKING_DIR, "train_features.parquet")
    df_schema = pd.read_parquet(train_parquet_path).head(0)
    feature_names = data_processor.get_feature_columns(df_schema.columns)

    # 3. Load Model and Threshold
    net = model.IPRVN(feature_names).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    thresh_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    if os.path.exists(thresh_path):
        best_thresh = float(np.load(thresh_path)[0])
        print(f"Loaded optimized threshold: {best_thresh}")
    else:
        best_thresh = 0.5
        print(f"Threshold file not found. Using default: {best_thresh}")

    # 4. Inference
    all_probs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            logits = net(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    y_probs = np.vstack(all_probs).flatten()
    y_pred = (y_probs >= best_thresh).astype(int)

    # 5. Create Submission
    submission = pd.DataFrame({"contact_id": df_test["contact_id"], "contact": y_pred})

    # Ensure directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission)}")
    print(
        f"Positive predictions: {submission['contact'].sum()} ({submission['contact'].mean():.4f})"
    )
