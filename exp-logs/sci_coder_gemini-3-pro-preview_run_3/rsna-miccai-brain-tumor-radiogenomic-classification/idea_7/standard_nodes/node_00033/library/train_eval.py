import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library import config, model as model_lib, data_loader


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(
            1
        )  # Ensure target shape matches output (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC
    # Handle edge case where only one class is present in the batch/set (though unlikely in val set)
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5

    if np.isnan(auc_score):
        auc_score = 0.5

    return epoch_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a list of (BraTS21ID, probability) tuples.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            probs_np = probs.cpu().numpy().flatten()

            # ids is a tuple of strings from the dataloader
            for pid, prob in zip(ids, probs_np):
                results.append((pid, prob))

    return results


def run_training(load_cached_data=True, debug_sample_size=None):
    """
    Main driver function to run the training pipeline.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader, _ = data_loader.get_dataloaders(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # Initialize Model
    model = model_lib.MGMTNet()
    model = model.to(device)

    # Optimizer and Loss
    # Note: Weight decay is explicitly 0.0 in config
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            counter += 1

        if counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(load_cached_data=True):
    """
    Loads the best model, predicts on test set, and saves submission file.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    # Get Test Loader
    _, _, test_loader = data_loader.get_dataloaders(load_cached_data=load_cached_data)

    # Load Model
    model = model_lib.MGMTNet()
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Run training first."
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)

    print("Generating predictions for test set...")
    predictions = predict(model, test_loader, device)

    # Create DataFrame
    df = pd.DataFrame(predictions, columns=["BraTS21ID", "MGMT_value"])

    # Ensure BraTS21ID is formatted correctly (though sample submission uses int, ids are strings here)
    # The sample submission provided in description shows:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # So we keep the string format or ensure it matches the competition requirement.
    # Usually keeping the ID as is from the test folder name is safest.

    # Save to CSV
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
