import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_roc_auc, get_pos_weights
from library.data import get_folds, BirdDataset, get_transforms, MixupCollate
from library.models import get_bird_model


def train_one_epoch(
    model, loader, optimizer, criterion, device, current_step, total_steps
):
    """
    Trains the model for one epoch.

    Returns:
        avg_loss (float): Average loss for the epoch.
        steps_taken (int): Number of optimization steps taken in this epoch.
    """
    model.train()
    running_loss = 0.0
    steps_in_epoch = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if current_step >= total_steps:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        current_step += 1
        steps_in_epoch += 1

    avg_loss = running_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
    return avg_loss, steps_in_epoch


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Returns:
        avg_loss (float): Average validation loss.
        auc_score (float): Macro-averaged ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auc_score = calculate_roc_auc(all_labels, all_preds)

    return avg_loss, auc_score


def run_training(load_cached_folds=True):
    """
    Main execution function for the Tri-Backbone Heterogeneous Ensemble training.

    Args:
        load_cached_folds (bool): Whether to load folds from cache.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Get Folds
    df_folds = get_folds(load_cached_data=load_cached_folds)

    # 2. Iterate Folds
    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} FOLD {fold}/{Config.N_FOLDS - 1} {'='*20}")

        # Split Data
        df_train = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
        df_val = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

        # Calculate Positive Weights for Loss (Class Imbalance Handling)
        pos_weights = get_pos_weights(df_train, device)
        # Apply factor if needed (default 1.0)
        pos_weights = pos_weights * Config.POS_WEIGHT_FACTOR

        # Prepare Datasets & Loaders
        train_dataset = BirdDataset(
            df_train, mode="train", transform=get_transforms(mode="train")
        )
        val_dataset = BirdDataset(
            df_val, mode="val", transform=get_transforms(mode="val")
        )

        # Use MixupCollate for training
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=MixupCollate(alpha=Config.MIXUP_ALPHA),
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Iterate Architectures (Heterogeneous Ensemble)
        for arch in Config.MODEL_ARCHS:
            print(f"\n--- Training Architecture: {arch} (Fold {fold}) ---")

            # Initialize Model
            model = get_bird_model(arch, pretrained=True)
            model.to(device)

            # Optimizer (Constant LR as per strategy)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Loss Function
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

            # Training Loop Variables
            best_auc = 0.0
            global_step = 0
            epoch = 0

            model_save_path = os.path.join(
                Config.WORKING_DIR, f"model_{arch}_fold_{fold}.pth"
            )

            # Train until TOTAL_STEPS is reached
            while global_step < Config.TOTAL_STEPS:
                epoch += 1

                train_loss, steps_taken = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    global_step,
                    Config.TOTAL_STEPS,
                )

                global_step += steps_taken

                # Validate
                val_loss, val_auc = validate(model, val_loader, criterion, device)

                print(
                    f"Epoch {epoch} | Step {global_step}/{Config.TOTAL_STEPS} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
                )

                # Save Best Model
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), model_save_path)
                    print(f"  >>> New Best AUC! Model saved to {model_save_path}")

            print(
                f"Finished training {arch} for fold {fold}. Best AUC: {best_auc:.10f}"
            )

            # Cleanup to save memory
            del model, optimizer, criterion
            torch.cuda.empty_cache()

    print("\nTraining Complete.")
