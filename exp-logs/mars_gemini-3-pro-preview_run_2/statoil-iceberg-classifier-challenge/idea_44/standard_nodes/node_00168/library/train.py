import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data_loader, model


def train_one_epoch(net, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels, _ in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (N, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = net(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy for monitoring
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(net, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles, labels, _ in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = net(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(debug_size=None):
    """
    Orchestrates the 5-Fold Cross-Validation training process.

    Args:
        debug_size (int, optional): Limit dataset size for debugging.
    """
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Starting training on device: {device}")
    print(f"Total Folds: {config.NUM_FOLDS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")

    fold_results = []

    for fold in range(config.NUM_FOLDS):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # 1. Data Loading
        train_loader, val_loader, _ = data_loader.get_loaders(
            fold_idx=fold, load_cached_data=True, debug_size=debug_size
        )

        # 2. Model Initialization
        net = model.InputAnchoredWideBodyNet().to(device)

        # 3. Optimizer & Criterion
        # Using Adam as specified (not AdamW)
        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

        # Scheduler: ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        criterion = nn.BCEWithLogitsLoss()

        # 4. Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(config.NUM_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                net, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(net, val_loader, criterion, device)

            # Step scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} - "
                f"Val Loss: {val_loss}, Val Acc: {val_acc} - "
                f"LR: {current_lr} - Time: {duration:.2f}s"
            )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Strictly preserve best weights using deepcopy
                best_model_state = copy.deepcopy(net.state_dict())
                print(f"New best model found! Val Loss: {best_val_loss}")
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # 5. Save Best Model for this Fold
        if best_model_state is not None:
            save_path = os.path.join(config.WORK_DIR, f"model_fold_{fold}.pth")
            print(f"Saving best model for fold {fold} to {save_path}")
            torch.save(best_model_state, save_path)
            fold_results.append(best_val_loss)
        else:
            print(f"Warning: No best model state found for fold {fold}.")

    print("\n" + "=" * 40)
    print("Training Complete.")
    print("Best Validation Losses per Fold:")
    for i, loss in enumerate(fold_results):
        print(f"Fold {i}: {loss}")
    print(f"Average Validation Loss: {np.mean(fold_results)}")
    print("=" * 40)
