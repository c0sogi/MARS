import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from library.config import Config, seed_everything
from library.utils import get_score
from library.data import get_dataloaders
from library.models import PawpularityModel


def train_one_epoch(model, optimizer, scheduler, dataloader, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, metadata, targets in dataloader:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, metadata)

        # Compute loss (BCEWithLogitsLoss expects logits)
        loss = criterion(logits.view(-1), targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            logits = model(images, metadata)
            loss = criterion(logits.view(-1), targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities in [0, 1]
            probs = torch.sigmoid(logits).view(-1)

            preds.append(probs.cpu().detach().numpy())
            valid_targets.append(targets.cpu().detach().numpy())

    epoch_loss = running_loss / dataset_size
    preds = np.concatenate(preds)
    valid_targets = np.concatenate(valid_targets)

    return epoch_loss, preds, valid_targets


def run_training():
    """
    Main training loop implementing 5-Fold CV for the ensemble.
    """
    seed_everything(Config.seed)
    Config.create_dirs()
    device = Config.device

    # Loss function for binary classification / regression in [0, 1]
    criterion = nn.BCEWithLogitsLoss()

    # Iterate over each architecture in the ensemble
    for model_name in Config.model_names:
        # Iterate over each fold
        for fold in range(Config.num_folds):
            print(f"\n{'='*40}")
            print(f"Training Model: {model_name} | Fold: {fold+1}/{Config.num_folds}")
            print(f"{'='*40}")

            # Get DataLoaders for this fold
            train_loader, val_loader = get_dataloaders(fold_idx=fold)

            # Initialize Model
            model = PawpularityModel(model_name=model_name, pretrained=True)
            model.to(device)

            # Optimizer with Differential Learning Rates
            # Separate backbone and head parameters
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

            optimizer_grouped_parameters = [
                # Backbone parameters (lower LR)
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "backbone" in n and not any(nd in n for nd in no_decay)
                    ],
                    "lr": Config.backbone_lr,
                    "weight_decay": Config.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "backbone" in n and any(nd in n for nd in no_decay)
                    ],
                    "lr": Config.backbone_lr,
                    "weight_decay": 0.0,
                },
                # Head parameters (higher LR)
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "head" in n and not any(nd in n for nd in no_decay)
                    ],
                    "lr": Config.lr,
                    "weight_decay": Config.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "head" in n and any(nd in n for nd in no_decay)
                    ],
                    "lr": Config.lr,
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

            # Scheduler: Cosine Annealing with Warmup
            num_train_steps = len(train_loader) * Config.epochs
            num_warmup_steps = len(train_loader) * Config.warmup_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps,
            )

            # Training Loop Variables
            best_rmse = float("inf")
            best_epoch = -1
            patience = 3  # Early stopping patience
            patience_counter = 0

            for epoch in range(Config.epochs):
                start_time = time.time()

                # Train
                train_loss = train_one_epoch(
                    model, optimizer, scheduler, train_loader, device, criterion
                )

                # Validate
                val_loss, val_preds, val_targets = valid_one_epoch(
                    model, val_loader, device, criterion
                )

                # Calculate RMSE (get_score handles scaling back to 100)
                val_rmse = get_score(val_targets, val_preds)

                elapsed = time.time() - start_time

                # Print metrics with full precision
                print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.2f}s")
                print(f"  Train Loss: {train_loss}")
                print(f"  Val Loss: {val_loss}")
                print(f"  Val RMSE: {val_rmse}")

                # Early Stopping Logic
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    patience_counter = 0

                    # Save Best Model
                    save_name = f"{model_name}_fold_{fold}.pth"
                    save_path = os.path.join(Config.working_dir, save_name)
                    torch.save(model.state_dict(), save_path)
                    print(f"  Best RMSE improved. Saved model to {save_path}")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            print(
                f"Fold {fold} finished. Best RMSE: {best_rmse} at epoch {best_epoch+1}"
            )

            # Cleanup to free GPU memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
