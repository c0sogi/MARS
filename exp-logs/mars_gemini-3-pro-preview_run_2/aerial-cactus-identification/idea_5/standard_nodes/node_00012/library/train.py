import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import CactusDataset, get_train_transforms, get_valid_transforms
from library.model import CustomSEResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to logits to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    epoch_loss = running_loss / dataset_size

    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    auc_score = calculate_roc_auc(all_targets, all_probs)

    return epoch_loss, auc_score


def run_training():
    """
    Orchestrates the training process for multiple seeds.
    """
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for seed in Config.SEEDS:
        print(f"\n{'='*20} Training Seed {seed} {'='*20}")
        seed_everything(seed)

        # --- Data Setup ---
        train_dataset = CactusDataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            transform=get_train_transforms(),
            mode="train",
            load_cached_data=True,
        )

        val_dataset = CactusDataset(
            metadata_path=Config.VAL_METADATA_PATH,
            transform=get_valid_transforms(),
            mode="val",
            load_cached_data=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # --- Model Setup ---
        model = CustomSEResNet(**Config.MODEL_PARAMS)
        model.to(device)

        # --- Optimization Setup ---
        criterion = getattr(nn, Config.LOSS_FN)()

        optimizer = getattr(optim, Config.OPTIMIZER_NAME)(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = getattr(optim.lr_scheduler, Config.SCHEDULER_NAME)(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # --- Training Loop ---
        best_auc = 0.0
        patience_counter = 0
        best_model_path = Config.get_model_path(seed)

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Step the scheduler
            scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping and Model Saving
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved for seed {seed} with AUC: {best_auc}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Finished training for seed {seed}. Best Val AUC: {best_auc}")
