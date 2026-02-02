import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import get_loaders
from library.models import get_model
from library.engine import train_one_epoch


def train_final_model(optimal_epochs):
    """
    Executes Phase 2: Production Training.
    Trains a fresh model on 100% of the data for the optimal number of epochs
    determined during the calibration phase.

    Args:
        optimal_epochs (int): The number of epochs to train.

    Returns:
        torch.nn.Module: The trained model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(
        f"Starting Production Phase: Training on 100% data for {optimal_epochs} epochs."
    )

    # 2. Prepare Class Weights
    # We need to calculate weights based on the full dataset (Train + Val)
    # This ensures consistency with the calibration phase
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Calculate weights (uses caching mechanism in utils)
    class_weights_np = calculate_class_weights(
        df_full, Config.TARGET_COLS, load_cached_data=True
    )
    class_weights = torch.tensor(class_weights_np).to(device)
    print(f"Global Class Weights: {class_weights_np}")

    # 3. Get Data Loader
    # mode='production' returns a single loader with all data and no validation set
    train_loader, _ = get_loaders(fold=0, mode="production")

    # 4. Initialize Model
    model = get_model(pretrained=True)

    # 5. Initialize Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 6. Initialize Scheduler
    # Crucial: T_0 is set to optimal_epochs to ensure the LR schedule completes
    # exactly at the end of training, matching the dynamics of the best fold in calibration.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=optimal_epochs, T_mult=1, eta_min=Config.MIN_LR
    )

    # 7. Define Loss Function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 8. Training Loop
    for epoch in range(optimal_epochs):
        print(f"\n[Production] Epoch {epoch + 1}/{optimal_epochs}")

        # Train one epoch
        # Note: train_one_epoch handles the scheduler step internally at the end of the epoch
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            scheduler=scheduler,
        )

        # We do not validate here as we are using all data for training.
        # The stopping point is fixed based on Phase 1 analysis.

    # 9. Save the Final Model
    save_path = os.path.join(Config.OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    return model
