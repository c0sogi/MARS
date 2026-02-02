import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.model import RoPESwiGLURMSNet
from library.data import get_dataloaders


def get_optimizer(model: nn.Module):
    """
    Configures the AdamW optimizer with parameter grouping.
    - Group 1: Weights (Linear, Embeddings) -> Weight Decay = 1e-2
    - Group 2: Biases, RMSNorm weights -> Weight Decay = 0.0
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Heuristic: Parameters with < 2 dimensions are usually biases or normalization gains (1D).
        # Linear weights and Embeddings are 2D or higher.
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": Config.WEIGHT_DECAY_WEIGHTS,
        },
        {
            "params": no_decay_params,
            "weight_decay": Config.WEIGHT_DECAY_BIASES,
        },
    ]

    optimizer = optim.AdamW(
        optim_groups,
        lr=Config.LEARNING_RATE,
    )
    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for x_cat, x_cont, y in loader:
        x_cat = x_cat.to(device, non_blocking=True)
        x_cont = x_cont.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cat, x_cont)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        count += y.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            all_preds.append(preds)
            all_targets.append(y)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        auc_score = compute_auc(all_targets, all_preds)
    else:
        auc_score = 0.5

    val_loss = running_loss / count if count > 0 else 0.0
    return val_loss, auc_score


def run_training(debug=False, load_cached_data=True):
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    print(f"Starting training run (Debug={debug})...")
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model
    model = RoPESwiGLURMSNet()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = get_optimizer(model)

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training Loop
    best_auc = 0.0

    # Adjust epochs if debugging
    epochs = 2 if debug else Config.EPOCHS

    print(f"Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")

    print(f"Training complete. Best Validation AUC: {best_auc:.15f}")
    return best_auc
