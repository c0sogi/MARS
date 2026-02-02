import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import LightweightPyramidNet


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape is (Batch, 1)

        optimizer.zero_grad()

        with amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs).float()
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    # Handle edge case if batch has only one class (unlikely for full val set)
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = 0.5

    return epoch_loss, auc


def run_training(
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 5,
    save_dir: str = "./working/idea_8",
):
    """
    Orchestrates the training process for a single model instance.
    """
    # 1. Setup
    set_seed(seed)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training for seed {seed} on {device}...")

    # 2. Data
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=batch_size, seed=seed)

    # 3. Model
    model = LightweightPyramidNet().to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Mixed Precision Scaler
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"model_seed_{seed}.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training finished for seed {seed}. Best Val AUC: {best_auc}")
    return best_model_path, best_auc
