import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.network import GroupedEfficientNet
from library.data_loader import get_dataloaders


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    Returns: average_loss (float), auc_score (float)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # BCE expects (N, 1)

        optimizer.zero_grad()

        # Forward pass (logits)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics tracking
        running_loss += loss.item() * inputs.size(0)

        # Convert logits to probabilities for AUC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.detach().cpu().numpy().flatten())
        all_preds.extend(probs.flatten())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch might have only one class (Cite debug_lesson_1)
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation.
    Returns: average_loss (float), auc_score (float)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy().flatten())
            all_preds.extend(probs.flatten())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch might have only one class (Cite debug_lesson_1)
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set using TTA (Original + HFlip + VFlip).
    Saves the result to output_path.
    """
    model.eval()
    predictions = []
    ids = []

    print("Generating submission with Test-Time Augmentation (TTA)...")

    with torch.no_grad():
        for inputs, subject_ids in test_loader:
            inputs = inputs.to(device)

            # 1. Original Prediction
            out_orig = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip Prediction (dim 3 is width)
            inputs_h = torch.flip(inputs, dims=[3])
            out_h = torch.sigmoid(model(inputs_h))

            # 3. Vertical Flip Prediction (dim 2 is height)
            inputs_v = torch.flip(inputs, dims=[2])
            out_v = torch.sigmoid(model(inputs_v))

            # Average predictions
            avg_preds = (out_orig + out_h + out_v) / 3.0

            predictions.extend(avg_preds.cpu().numpy().flatten())
            ids.extend(subject_ids.numpy())

    # Write to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("BraTS21ID,MGMT_value\n")
        for pid, pred in zip(ids, predictions):
            # Format ID as 5-digit string (e.g., 00001)
            f.write(f"{int(pid):05d},{pred}\n")

    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main driver function for training, validation, and submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    print(f"Initializing Model: {Config.BACKBONE}...")
    model = GroupedEfficientNet().to(device)

    # 4. Optimizer & Loss
    # AdamW with specific learning rate and weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Binary Cross Entropy with Logits (Numerical stability)
    # No label smoothing as per requirements
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting Training Loop...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} [{duration:.0f}s] - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc} - "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save Best Model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New Best Model Saved! AUC: {best_val_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")

    # 6. Generate Submission
    # Load best model weights
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
