import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import EfficientNetExpert


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Calculate AUC. Handle edge case where only one class is present in batch (unlikely in fold)
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def run_expert_training(load_cached_data=True):
    """
    Main driver function to train the model across all 5 Folds.
    Saves the best model for each Fold.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Starting Training on device: {device}")

    # Iterate over each Expert (Anatomical Plane)
    for expert_name, offset in Config.EXPERTS.items():
        print(f"\n{'='*20}")
        print(f" Training {expert_name} (Offset: {offset})")
        print(f"{'='*20}")

        # 5-Fold Cross Validation
        for fold in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold}/{Config.NUM_FOLDS - 1} ---")

            # 1. Reproducibility
            set_seed(Config.SEED + fold)

            # 2. Data Loading
            train_loader, val_loader = get_dataloaders(
                fold_idx=fold,
                expert_offset=offset,
                load_cached_data=load_cached_data,
            )

            # 3. Model Initialization
            model = EfficientNetExpert(pretrained=True)
            model = model.to(device)

            # 4. Optimizer & Loss
            # AdamW with weight decay as specified in Config
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            criterion = nn.BCEWithLogitsLoss()

            # 5. Training Loop with Early Stopping
            best_val_auc = 0.0
            patience_counter = 0
            best_model_path = os.path.join(
                Config.WORK_DIR, f"best_model_{expert_name}_fold{fold}.pth"
            )

            for epoch in range(Config.NUM_EPOCHS):
                start_time = time.time()

                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                val_loss, val_auc = validate(model, val_loader, criterion, device)

                elapsed = time.time() - start_time

                print(
                    f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                    f"Time: {elapsed:.1f}s | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val AUC: {val_auc:.16f}"
                )

                # Checkpoint Logic (Maximize AUC)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  [+] New Best AUC! Model saved to {best_model_path}")
                else:
                    patience_counter += 1
                    print(
                        f"  [-] No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                    )

                # Early Stopping
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Clean up to save memory
            del model, optimizer, criterion, train_loader, val_loader
            torch.cuda.empty_cache()

    print("\nAll training tasks completed.")
