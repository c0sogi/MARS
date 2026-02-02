import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed
from library.model import LateFusionSwishGatedResFunnel
from library.data_loader import get_dataloaders


def get_optimizer(model):
    """
    Configures the AdamW optimizer with decoupled weight decay.
    Group 1: Decay 1e-2 (Weights of Linear, Embedding, Attention)
    Group 2: Decay 0.0 (Biases, LayerNorm, Positional Embeddings)
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify parameters that should not have weight decay
        # 1. Dimensions < 2 (usually biases)
        # 2. Explicitly named 'bias'
        # 3. LayerNorm parameters ('norm' in name)
        # 4. Positional Embeddings ('pos_embed' in name)
        if param.dim() < 2 or "bias" in name or "norm" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP1},
        {"params": no_decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP2},
    ]

    optimizer = optim.AdamW(param_groups, lr=Config.LR)
    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for x_cont, x_cat, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cont, x_cat).squeeze()
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_cont.size(0)
        num_samples += x_cont.size(0)

    return total_loss / num_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_cont, x_cat).squeeze()
            loss = criterion(logits, y)

            total_loss += loss.item() * x_cont.size(0)
            num_samples += x_cont.size(0)

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / num_samples
    auc = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc


def run_training(data_fraction=1.0, epochs=Config.EPOCHS):
    """
    Main training function.

    Args:
        data_fraction (float): Fraction of data to use (for debugging).
        epochs (int): Number of epochs to train.
    """
    set_seed(Config.SEED)

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, data_fraction=data_fraction
    )

    # Initialize Model
    model = LateFusionSwishGatedResFunnel().to(Config.DEVICE)

    # Optimizer and Scheduler
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")
    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step()

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_auc}")

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc
