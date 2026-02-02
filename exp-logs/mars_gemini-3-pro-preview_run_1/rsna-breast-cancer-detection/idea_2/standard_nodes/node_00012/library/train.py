import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet, probabilistic_f1, predict_and_submit


def train_epoch(model, loader, optimizer, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()

    total_loss_meter = 0.0
    cancer_loss_meter = 0.0
    density_loss_meter = 0.0

    # Loss functions
    # Weighted BCE for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    cancer_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # CrossEntropy for density (ignore -1 for missing labels)
    density_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    scaler = GradScaler()

    for batch_idx, (inputs, cancer_targets, density_targets) in enumerate(loader):
        inputs = inputs.to(device)
        cancer_targets = cancer_targets.to(device).unsqueeze(1)  # [B, 1]
        density_targets = density_targets.to(device)  # [B]

        optimizer.zero_grad()

        # Forward
        with autocast():
            cancer_logits, density_logits = model(inputs)

            # Compute Losses
            loss_c = cancer_criterion(cancer_logits, cancer_targets)
            loss_d = density_criterion(density_logits, density_targets)

            # Weighted Sum
            loss = loss_c + Config.AUX_WEIGHT * loss_d

        # Backward
        scaler.scale(loss).backward()

        # No gradient clipping as per lesson 00009

        scaler.step(optimizer)
        scaler.update()

        # Update meters
        total_loss_meter += loss.item()
        cancer_loss_meter += loss_c.item()
        density_loss_meter += loss_d.item()

    avg_loss = total_loss_meter / len(loader)
    avg_cancer_loss = cancer_loss_meter / len(loader)
    avg_density_loss = density_loss_meter / len(loader)

    print(
        f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} (Cancer: {avg_cancer_loss:.4f}, Density: {avg_density_loss:.4f})"
    )
    return avg_loss


def validate_epoch(model, loader, device):
    """
    Executes validation and returns pF1 score.
    """
    model.eval()

    all_cancer_probs = []
    all_cancer_targets = []

    all_density_preds = []
    all_density_targets = []

    with torch.no_grad():
        for inputs, cancer_targets, density_targets in loader:
            inputs = inputs.to(device)

            with autocast():
                cancer_logits, density_logits = model(inputs)

            # Cancer Probs
            probs = torch.sigmoid(cancer_logits).cpu().float().numpy().flatten()
            all_cancer_probs.extend(probs)
            all_cancer_targets.extend(cancer_targets.numpy())

            # Density Preds
            preds_d = torch.argmax(density_logits, dim=1).cpu().numpy()
            all_density_preds.extend(preds_d)
            all_density_targets.extend(density_targets.numpy())

    # Metrics
    all_cancer_probs = np.array(all_cancer_probs)
    all_cancer_targets = np.array(all_cancer_targets)

    pf1 = probabilistic_f1(all_cancer_probs, all_cancer_targets)

    # Filter density targets for accuracy (remove -1)
    all_density_preds = np.array(all_density_preds)
    all_density_targets = np.array(all_density_targets)
    valid_mask = all_density_targets != -1
    if valid_mask.sum() > 0:
        density_acc = accuracy_score(
            all_density_targets[valid_mask], all_density_preds[valid_mask]
        )
    else:
        density_acc = 0.0

    # Print full precision without formatting as requested
    print(f"Validation | pF1: {pf1} | Density Acc: {density_acc}")

    return pf1


def train_model(
    epochs=Config.EPOCHS,
    debug=Config.DEBUG,
    load_cached_data=True,
    save_path=Config.MODEL_SAVE_PATH,
    submission_path=Config.SUBMISSION_PATH,
):
    """
    Main training loop with Early Stopping and Inference.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Model
    print(f"Initializing Model: {Config.BACKBONE}")
    model = MultiTaskEfficientNet(Config.BACKBONE, pretrained=True)
    model.to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)

    # 4. Training Loop
    best_pf1 = -1.0
    patience = 4
    patience_counter = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device, epoch)
        val_pf1 = validate_epoch(model, val_loader, device)

        scheduler.step()

        # Save Best
        if val_pf1 > best_pf1:
            print(f"New Best pF1: {val_pf1} (was {best_pf1}). Saving model.")
            best_pf1 = val_pf1
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Warning: No model saved. Using current weights.")

    predict_and_submit(model, test_loader, device, submission_path)
