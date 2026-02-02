import os
import gc
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import RetinopathyDataset, get_transforms, load_dataframe
from library.models import RetinopathyModel
from library.engine import train_loop


def run_training(
    model_name: str,
    epochs: int = Config.epochs,
    debug: bool = Config.debug,
    debug_samples: int = Config.debug_samples,
):
    """
    Runs Stratified K-Fold Cross-Validation training for a specific model architecture.

    Args:
        model_name (str): Name of the model architecture (e.g., 'tf_efficientnet_b5_ns').
        epochs (int): Number of training epochs per fold.
        debug (bool): If True, runs on a small subset of data.
        debug_samples (int): Number of samples to use in debug mode.
    """
    seed_everything(Config.seed)

    print(f"=== Starting Training for Architecture: {model_name} ===")

    # 1. Load Data
    # We load both train and val metadata and combine them for K-Fold CV
    # load_dataframe handles caching internally
    train_df = load_dataframe(Config.train_csv_path, "train_df")
    val_df = load_dataframe(Config.val_csv_path, "val_df")

    # Concatenate to form the full development set
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Debug Mode: Subset data if requested
    if debug:
        print(f"Debug mode active. Using {debug_samples} samples.")
        full_df = full_df.head(debug_samples)

    # 2. Stratified K-Fold
    # We use the 'diagnosis' column for stratification
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    y = full_df["diagnosis"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y)):
        print(f"\n--- Fold {fold + 1}/{Config.n_folds} ---")

        # Split Data
        df_train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = full_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = RetinopathyDataset(
            df_train_fold, transform=get_transforms("train"), mode="train"
        )
        val_dataset = RetinopathyDataset(
            df_val_fold, transform=get_transforms("valid"), mode="val"
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Initialize Model
        model = RetinopathyModel(model_name=model_name, pretrained=True)
        model = model.to(Config.device)

        # Loss Function (MSE for Regression formulation)
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=Config.min_lr
        )

        # Checkpoint Path
        # Saves to ./working/idea_3/{model_name}_fold_{fold}.pth
        save_path = os.path.join(Config.working_dir, f"{model_name}_fold_{fold}.pth")

        # Run Training Loop
        train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.device,
            epochs=epochs,
            save_path=save_path,
        )

        # Cleanup to free GPU memory for the next fold
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        gc.collect()
        torch.cuda.empty_cache()


def train_all_models():
    """
    Iterates through all architectures defined in Config and trains them sequentially.
    """
    for arch in Config.model_archs:
        run_training(arch)
