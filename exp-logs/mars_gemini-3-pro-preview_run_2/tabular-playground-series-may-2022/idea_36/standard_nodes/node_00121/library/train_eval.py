import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import load_and_preprocess_data
from library.model import ModalityScaledHybridSwiGLU


def get_optimizer(model):
    """
    Constructs the AdamW optimizer with strict decoupled weight decay.

    Group 1 (Decay): Weights of Linear, Embeddings, Attention.
    Group 2 (No Decay): Biases, LayerNorms, Positional Embeddings, Modality Scalars.
    """
    decay_params = []
    no_decay_params = []

    # Iterate through named parameters to categorize them
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Conditions for No Decay
        if (
            "bias" in name
            or "norm" in name
            or "pos_encoder" in name
            or "lambda_" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {
            "params": decay_params,
            "weight_decay": Config.WEIGHT_DECAY_GROUP1,
        },
        {
            "params": no_decay_params,
            "weight_decay": Config.WEIGHT_DECAY_GROUP2,
        },
    ]

    optimizer = optim.AdamW(optimizer_groups, lr=Config.LEARNING_RATE)

    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        continuous = batch["continuous"].to(device, non_blocking=True)
        sequence = batch["sequence"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(continuous, sequence)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * continuous.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            continuous = batch["continuous"].to(device, non_blocking=True)
            sequence = batch["sequence"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(continuous, sequence)
            loss = criterion(logits, targets)

            running_loss += loss.item() * continuous.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            continuous = batch["continuous"].to(device, non_blocking=True)
            sequence = batch["sequence"].to(device, non_blocking=True)

            logits = model(continuous, sequence)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def run_training():
    """
    Main execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = load_and_preprocess_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = ModalityScaledHybridSwiGLU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_auc = 0.0
    patience = (
        5  # Early stopping patience (not strictly defined in config but good practice)
    )
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 7. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # 8. Submission
    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Ensure lengths match
    if len(predictions) != len(test_meta):
        raise ValueError(
            f"Prediction length {len(predictions)} does not match metadata length {len(test_meta)}"
        )

    submission_df = pd.DataFrame(
        {"id": test_meta["id"], "target": predictions.flatten()}
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
