import os
import heapq
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.data import get_dataloaders, mixup_data, mixup_criterion
from library.models import get_model
from library.optimization import get_optimizer
from library.utils import seed_everything, calculate_roc_auc, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training with Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        # alpha=1.0 is standard for Mixup (samples from Beta(1,1) -> Uniform)
        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)

        optimizer.zero_grad()
        outputs = model(images)

        # Mixup Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and robust ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate robust ROC AUC
    epoch_auc = calculate_roc_auc(all_labels, all_preds)

    return epoch_loss, epoch_auc


def train_fold(config: Config, fold: int, model_name: str):
    """
    Trains a specific architecture for a specific fold.
    Implements Intra-Fold Snapshotting to save the Top-K best checkpoints.

    Args:
        config (Config): Configuration object.
        fold (int): Fold index (used for seeding and file naming).
        model_name (str): Name of the architecture to train.
    """
    # Ensure reproducibility
    seed_everything(config.SEED + fold)
    device = get_device()

    # Load Data (using cached data if available)
    # The loader returns the fold-specific split
    train_loader, val_loader, _, _ = get_dataloaders(
        config, fold=fold, load_cached_data=True
    )

    # Initialize Model
    model = get_model(model_name, config, pretrained=True)

    # Initialize Optimizer (Lookahead wrapping AdamW)
    optimizer = get_optimizer(model, config)

    # Initialize Scheduler (Cosine Annealing)
    # We pass the Lookahead optimizer wrapper; it exposes param_groups correctly.
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Snapshotting State
    # Heap stores tuples of (auc, epoch, checkpoint_path)
    # We use a min-heap to keep track of the Top-K.
    # If the heap grows beyond K, we pop the smallest AUC (the worst of the best).
    top_k_checkpoints = []

    # Base directory for this model
    # We call get_checkpoint_path once to ensure directory creation
    _ = config.get_checkpoint_path(model_name, fold, rank=0)
    base_dir = os.path.join(config.CHECKPOINT_DIR, model_name)

    print(f"Starting training for {model_name} - Fold {fold}")
    print(
        f"Training for {config.NUM_EPOCHS} epochs. Saving Top-{config.TOP_K_CHECKPOINTS} snapshots."
    )

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Training Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Snapshot Logic
        # 1. Save current model to a temporary file
        temp_filename = f"temp_{model_name}_fold_{fold}_epoch_{epoch}.pth"
        temp_path = os.path.join(base_dir, temp_filename)
        torch.save(model.state_dict(), temp_path)

        # 2. Add to heap
        # We push (auc, epoch, path). Python compares tuples element-wise.
        heapq.heappush(top_k_checkpoints, (val_auc, epoch, temp_path))

        # 3. Prune if necessary
        if len(top_k_checkpoints) > config.TOP_K_CHECKPOINTS:
            # Remove the item with the lowest AUC from the heap
            worst_auc, worst_epoch, worst_path = heapq.heappop(top_k_checkpoints)

            # Delete the corresponding file from disk to save space
            if os.path.exists(worst_path):
                os.remove(worst_path)

    # Finalization: Rename the surviving Top-K checkpoints
    print(
        f"\nFinalizing Top {config.TOP_K_CHECKPOINTS} checkpoints for {model_name} Fold {fold}:"
    )

    # Sort by AUC descending (Best first)
    sorted_checkpoints = sorted(top_k_checkpoints, key=lambda x: x[0], reverse=True)

    for rank, (auc, epoch, path) in enumerate(sorted_checkpoints):
        # Generate the standardized final path (e.g., ..._rank_0.pth)
        final_path = config.get_checkpoint_path(model_name, fold, rank=rank)

        if os.path.exists(path):
            # If the final path exists (e.g. from a previous run), overwrite it
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(path, final_path)

        print(
            f"Rank {rank}: Epoch {epoch}, AUC {auc:.10f} -> {os.path.basename(final_path)}"
        )
