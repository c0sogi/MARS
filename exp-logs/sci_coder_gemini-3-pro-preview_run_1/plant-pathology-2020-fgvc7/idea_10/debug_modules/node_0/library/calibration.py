import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc

from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import get_loaders
from library.models import get_model
from library.engine import train_one_epoch, validate_one_epoch


def run_calibration(epochs=None):
    """
    Executes Phase 1: Hyperparameter Calibration.
    Runs Stratified K-Fold CV to determine the optimal number of training epochs.

    Args:
        epochs (int, optional): Override the number of epochs from Config.

    Returns:
        int: The optimal number of epochs (E_opt).
    """
    # 1. Setup
    seed_everything(Config.SEED)
    num_epochs = epochs if epochs is not None else Config.EPOCHS_CALIBRATION
    device = Config.DEVICE

    print(
        f"Starting Calibration Phase: {Config.N_FOLDS} Folds, {num_epochs} Epochs per fold."
    )

    # 2. Prepare Class Weights
    # Load all available training data to calculate global class weights
    # We use the metadata files defined in Config
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Calculate weights (cached internally by utils)
    class_weights_np = calculate_class_weights(
        df_full, Config.TARGET_COLS, load_cached_data=True
    )
    class_weights = torch.tensor(class_weights_np).to(device)
    print(f"Global Class Weights: {class_weights_np}")

    # 3. Calibration Loop
    # We track validation AUC for every epoch across all folds
    # Shape: [Fold, Epoch]
    fold_auc_history = np.zeros((Config.N_FOLDS, num_epochs))

    for fold in range(Config.N_FOLDS):
        print(f"\n[Calibration] Fold {fold + 1}/{Config.N_FOLDS}")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_loaders(fold=fold, mode="calibration")

        # Initialize Model
        model = get_model(pretrained=True)

        # Initialize Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Initialize Scheduler (Cosine Annealing synchronized with num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=Config.MIN_LR
        )

        # Define Loss Function
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training Loop for this Fold
        for epoch in range(num_epochs):
            # Train
            train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                scheduler=scheduler,
            )

            # Validate
            val_loss, val_auc = validate_one_epoch(
                model=model, data_loader=val_loader, device=device, criterion=criterion
            )

            # Record Metric
            fold_auc_history[fold, epoch] = val_auc

            print(
                f"Fold {fold+1} Epoch {epoch+1} - Train Loss: {train_loss} Val Loss: {val_loss} Val AUC: {val_auc}"
            )

        # Cleanup to free GPU memory for next fold
        del model, optimizer, scheduler, criterion, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Analysis
    print("\n[Calibration] Analysis of Mean Validation AUC per Epoch:")

    # Calculate mean AUC across folds for each epoch
    mean_auc_per_epoch = np.mean(fold_auc_history, axis=0)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}: Mean AUC = {mean_auc_per_epoch[epoch]}")

    # Identify Global Optimal Epoch (E_opt)
    # argmax returns index (0-based), we convert to epoch count (1-based)
    best_epoch_idx = np.argmax(mean_auc_per_epoch)
    optimal_epochs = best_epoch_idx + 1
    best_auc = mean_auc_per_epoch[best_epoch_idx]

    print(
        f"\n[Calibration] Optimal Epochs determined: {optimal_epochs} (Peak Mean AUC: {best_auc})"
    )

    return optimal_epochs
