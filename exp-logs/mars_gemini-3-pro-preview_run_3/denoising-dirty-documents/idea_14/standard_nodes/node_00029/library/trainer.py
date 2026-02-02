import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.network import DnCNN
from library.data_loader import DenoisingDataset, load_dataset_patches


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Calculate ground truth noise: Noise = Noisy_Input - Clean_Target
        # The network predicts the noise residual R(x)
        noise_target = inputs - targets

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, noise_target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Executes validation loop.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Ground truth noise
            noise_target = inputs - targets

            outputs = model(inputs)

            loss = criterion(outputs, noise_target)
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def train_ensemble_member(member_id, seed):
    """
    Trains a single ZI-ResDnCNN model using the curriculum strategy.

    Args:
        member_id (int): Identifier for the ensemble member.
        seed (int): Random seed for reproducibility.
    """
    # Set seed for reproducibility
    seed_everything(seed)

    device = torch.device(Config.DEVICE)
    print(f"Initializing training for Member {member_id} with seed {seed} on {device}")

    # Initialize Model
    model = DnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_RES_BLOCKS,
    ).to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # Load Validation Data (Shared across stages)
    print("Loading Validation Data...")
    val_patches, val_targets = load_dataset_patches("val", load_cached_data=True)
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Variables to track best model
    best_val_loss = float("inf")
    best_model_state = None

    # Curriculum Stages
    stages = [
        {
            "name": "Stage 1 (Sparse)",
            "mode": "train_sparse",
            "epochs": Config.STAGE_1_EPOCHS,
        },
        {
            "name": "Stage 2 (Dense)",
            "mode": "train_dense",
            "epochs": Config.STAGE_2_EPOCHS,
        },
    ]

    for stage in stages:
        stage_name = stage["name"]
        print(f"\n--- Starting {stage_name} ---")

        # Load Data for current stage
        train_patches, train_targets = load_dataset_patches(
            stage["mode"], load_cached_data=True
        )
        train_dataset = DenoisingDataset(
            train_patches, train_targets, augment=Config.USE_AUGMENTATION
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Scheduler for this stage
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage["epochs"]
        )

        # If we have a best model from previous stage, ensure we start from there
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        patience_counter = 0

        for epoch in range(1, stage["epochs"] + 1):
            start_time = time.time()

            # Training Step
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validation Step
            val_loss = validate(model, val_loader, criterion, device)

            # Scheduler Step
            scheduler.step()

            elapsed_time = time.time() - start_time
            val_rmse = np.sqrt(val_loss)

            print(
                f"Epoch {epoch}/{stage['epochs']} [{stage_name}] | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val RMSE: {val_rmse:.10f} | "
                f"Time: {elapsed_time:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered in {stage_name} at epoch {epoch}")
                break

        # Clean up memory
        del train_patches, train_targets, train_dataset, train_loader
        import gc

        gc.collect()

    # Save the best model
    if best_model_state is not None:
        save_path = os.path.join(Config.WORKING_DIR, f"model_{member_id}.pth")
        torch.save(best_model_state, save_path)
        print(
            f"\nMember {member_id} training complete. Best Val RMSE: {np.sqrt(best_val_loss):.10f}"
        )
        print(f"Model saved to {save_path}")
    else:
        print(f"Member {member_id} training failed to produce a valid model.")
