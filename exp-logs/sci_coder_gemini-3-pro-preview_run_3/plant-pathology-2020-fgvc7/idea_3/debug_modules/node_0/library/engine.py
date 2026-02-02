import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import Config
from library.utils import compute_metric, save_checkpoint
from library.data import get_loaders


class BNLoaderWrapper:
    """
    Wrapper for DataLoader to yield only images for torch.optim.swa_utils.update_bn.
    The standard update_bn expects the loader to yield input tensors or (input, target) tuples,
    but our dataset yields a dictionary.
    """

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield batch["image"]


def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        # Dataset returns float one-hot vectors.
        # CrossEntropyLoss expects class indices (LongTensor).
        target_indices = torch.argmax(targets, dim=1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, target_indices)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and targets for metric calculation
        # Apply softmax to logits to get probabilities
        probs = torch.softmax(outputs, dim=1)
        all_targets.append(targets.cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_metric = compute_metric(all_targets, all_preds)

    return epoch_loss, epoch_metric


def validate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            target_indices = torch.argmax(targets, dim=1)

            outputs = model(images)
            loss = criterion(outputs, target_indices)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_metric = compute_metric(all_targets, all_preds)

    return epoch_loss, epoch_metric


def fit(model, optimizer, device, model_name, debug=False):
    """
    Main training loop handling Progressive Resizing, SWA, and Early Stopping.
    """
    # Setup Loss with Class Weights
    weights = Config.CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Progressive Resizing Configuration
    current_img_size = Config.MODEL_2_IMG_SIZE
    resize_epoch = -1
    target_img_size = -1

    if model_name == Config.MODEL_1_NAME:
        current_img_size = Config.MODEL_1_START_IMG_SIZE
        target_img_size = Config.MODEL_1_IMG_SIZE
        resize_epoch = Config.MODEL_1_RESIZE_EPOCH

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_loaders(
        current_img_size, batch_size=Config.BATCH_SIZE, debug=debug
    )

    # SWA Initialization
    swa_model = None
    swa_scheduler = None
    if Config.USE_SWA:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    # Standard Scheduler (Cosine Annealing)
    # Runs until SWA starts
    steps_before_swa = Config.SWA_START_EPOCH if Config.USE_SWA else Config.EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=steps_before_swa)

    best_metric = -1.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {model_name}...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Handle Progressive Resizing
        # If we just finished the resize_epoch, switch to the larger size
        if model_name == Config.MODEL_1_NAME and epoch == resize_epoch + 1:
            print(
                f"Progressive Resizing: Switching input size from {current_img_size} to {target_img_size} at epoch {epoch}"
            )
            current_img_size = target_img_size
            train_loader, val_loader, _ = get_loaders(
                current_img_size, batch_size=Config.BATCH_SIZE, debug=debug
            )

        # Train and Validate
        train_loss, train_metric = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        val_loss, val_metric = validate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss} | Train AUC: {train_metric} | Val Loss: {val_loss} | Val AUC: {val_metric}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_metric,
                },
                filename=f"best_model_{model_name}.pth",
            )
        else:
            patience_counter += 1

        # SWA Logic
        if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Early Stopping
        # We only stop early if we are not in the critical SWA phase or if performance is degrading significantly
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Finalize SWA
    if Config.USE_SWA and swa_model is not None:
        print("Finalizing SWA Model...")
        # Update Batch Normalization statistics
        bn_loader = BNLoaderWrapper(train_loader)
        update_bn(bn_loader, swa_model, device=device)

        # Validate SWA Model
        swa_val_loss, swa_val_metric = validate(
            swa_model, val_loader, device, criterion
        )
        print(
            f"SWA Final Results | Val Loss: {swa_val_loss} | Val AUC: {swa_val_metric}"
        )

        # Save SWA Model
        save_checkpoint(
            {
                "epoch": Config.EPOCHS,
                "model_state_dict": swa_model.state_dict(),
                "val_metric": swa_val_metric,
            },
            filename=f"swa_model_{model_name}.pth",
        )

        return swa_model

    return model
