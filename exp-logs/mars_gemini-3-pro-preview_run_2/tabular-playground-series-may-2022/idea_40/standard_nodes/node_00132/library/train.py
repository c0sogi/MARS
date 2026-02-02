import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HybridNetwork


def setup_optimizer(model):
    """
    Sets up AdamW optimizer with strict decoupled weight decay.
    Group 1 (Decay 1e-2): Weights of Linear, Embeddings, Attention.
    Group 2 (Decay 0.0): Biases, LayerNorm, BatchNorm, Positional Embeddings.
    """
    decay_params = []
    no_decay_params = []

    # Identify parameters to exclude from decay
    # 1. Positional Embeddings (explicitly named 'pos_embed')
    # 2. Biases (name ends with '.bias')
    # 3. Normalization weights (LayerNorm/BatchNorm weights usually named 'weight' inside a norm module)
    #    We check for 'norm' in the name and '.weight' suffix as a heuristic for this specific architecture.

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "pos_embed" in name:
            no_decay_params.append(param)
        elif name.endswith(".bias"):
            no_decay_params.append(param)
        elif ("norm" in name or "bn" in name) and name.endswith(".weight"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP1},
        {"params": no_decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP2},
    ]

    optimizer = optim.AdamW(optim_groups, lr=Config.LEARNING_RATE)
    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    dataset_size = len(loader.dataset)

    for x_cat, x_cont, targets in loader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns probabilities (sigmoid applied)
        preds = model(x_cat, x_cont)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_cat.size(0)

    return total_loss / dataset_size


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for x_cat, x_cont, targets in loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            targets = targets.to(device)

            preds = model(x_cat, x_cont)
            loss = criterion(preds, targets)

            total_loss += loss.item() * x_cat.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    # Handle potential edge cases where only one class is present
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont, _ in loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            preds = model(x_cat, x_cont)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main training execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # 3. Model
    print("Initializing Model...")
    model = HybridNetwork().to(Config.DEVICE)

    # 4. Optimization
    optimizer = setup_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )
    # Model outputs probabilities, so we use BCELoss
    criterion = nn.BCELoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6e} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc + Config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epoch+1} epochs. Best AUC: {best_auc:.10f}"
            )
            break

    # 6. Submission
    print("Loading best model for inference...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    print("Generating submission...")
    test_probs = predict(model, test_loader, Config.DEVICE)

    # Load Test Metadata to ensure correct ID mapping
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    submission_df = pd.DataFrame({"id": test_meta["id"], "target": test_probs})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
