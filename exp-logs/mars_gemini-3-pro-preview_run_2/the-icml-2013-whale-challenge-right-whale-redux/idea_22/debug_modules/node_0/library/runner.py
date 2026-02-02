import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import get_dataloaders
from library.models import WhaleModel
from library.engine import train_model
from library.utils import seed_everything


def run_fold(fold_idx, model_name, load_cached_data=True):
    """
    Executes the training pipeline for a single fold.
    Implements Multi-Objective Checkpointing by saving distinct models for
    Best AUC (Discriminator) and Best Loss (Calibrator).

    Args:
        fold_idx (int): The index of the current fold. Used for seeding and file naming.
        model_name (str): The name of the model architecture (e.g., 'tf_efficientnet_b0_ns').
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
    """
    # 1. Set Random Seed for Reproducibility
    # We offset the seed by fold_idx to ensure different initialization/batching if running multiple times
    seed_everything(Config.seed + fold_idx)

    print(f"--- Starting Training | Fold: {fold_idx} | Model: {model_name} ---")

    # 2. Data Loading
    # Uses the fixed stratified split defined in metadata/train.csv and metadata/val.csv
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    device = Config.device
    model = WhaleModel(model_name=model_name, pretrained=Config.pretrained)
    model = model.to(device)

    # 4. Optimizer & Scheduler Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    # 5. Define Checkpoint Paths
    # Ensure the checkpoint directory exists
    checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define paths for both objectives
    save_path_auc = os.path.join(
        checkpoint_dir, f"{model_name}_fold_{fold_idx}_best_auc.pth"
    )
    save_path_loss = os.path.join(
        checkpoint_dir, f"{model_name}_fold_{fold_idx}_best_loss.pth"
    )

    # 6. Execute Training Loop
    # The engine handles the logic for saving to save_path_auc when AUC improves
    # and save_path_loss when Loss improves.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=Config.patience,
        save_path_auc=save_path_auc,
        save_path_loss=save_path_loss,
    )

    print(f"Fold {fold_idx} completed.")
    print(f"Best AUC Checkpoint: {save_path_auc}")
    print(f"Best Loss Checkpoint: {save_path_loss}")
