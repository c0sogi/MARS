import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import CMTSINModel
from library.losses import MultiTaskLoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pf1(preds, targets, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score.

    Args:
        preds (torch.Tensor): Predicted probabilities (N,).
        targets (torch.Tensor): Binary ground truth labels (N,).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    # Ensure inputs are flat
    preds = preds.view(-1)
    targets = targets.view(-1)

    # pTP: Sum of probabilities for positive ground truth instances
    # pTP = Sum(pred_i * target_i)
    p_tp = torch.sum(preds * targets)

    # pFP: Sum of probabilities for negative ground truth instances
    # pFP = Sum(pred_i * (1 - target_i))
    p_fp = torch.sum(preds * (1 - targets))

    # Total Positives (TP + FN) = Sum(targets)
    total_positives = torch.sum(targets)

    # Probabilistic Precision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(preds * targets + preds - preds * targets) = Sum(preds)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # Probabilistic Recall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # pF1 = 2 * (pPrec * pRec) / (pPrec + pRec)
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1.item()


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # Iterate over batches
    # Using simple iteration without tqdm to keep logs clean as requested,
    # or minimal logging.

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)

        targets = {}
        for k, v in batch["targets"].items():
            targets[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            preds = model(images, meta)
            loss_dict = loss_fn(preds, targets)
            loss = loss_dict["total_loss"]

        # Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    all_cancer_preds = []
    all_cancer_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)

            targets = {}
            for k, v in batch["targets"].items():
                targets[k] = v.to(device, non_blocking=True)

            # Forward pass (no autocast needed for eval usually, but consistent with train)
            # Using float32 for precision in validation
            preds = model(images, meta)

            loss_dict = loss_fn(preds, targets)
            running_loss += loss_dict["total_loss"].item()

            # Collect predictions for pF1 calculation
            # Primary cancer head outputs logits, apply sigmoid
            cancer_probs = torch.sigmoid(preds["cancer"])
            all_cancer_preds.append(cancer_probs.cpu())
            all_cancer_targets.append(targets["cancer"].cpu())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_cancer_preds = torch.cat(all_cancer_preds)
    all_cancer_targets = torch.cat(all_cancer_targets)

    # Compute Metric
    pf1 = calculate_pf1(all_cancer_preds, all_cancer_targets)

    return avg_loss, pf1


def run_training():
    """
    Main function to orchestrate training, validation, and saving.
    """
    print("Initializing CMT-SIN Training...")

    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data
    print("Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    print(f"Creating Model: {Config.MODEL_NAME}")
    model = CMTSINModel()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    loss_fn = MultiTaskLoss()
    loss_fn.to(device)

    scaler = GradScaler()

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, loss_fn, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val pF1:    {val_pf1:.9f}")  # Full precision

        # Checkpoint & Early Stopping
        if val_pf1 > best_pf1:
            print(f"  [+] New Best pF1! Saving model to {Config.MODEL_SAVE_PATH}")
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation pF1: {best_pf1:.9f}")
