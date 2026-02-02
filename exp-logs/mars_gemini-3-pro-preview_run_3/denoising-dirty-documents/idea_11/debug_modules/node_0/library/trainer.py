import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.network import CAResDnCNN
from library.dataset import DenoisingDataset, extract_patches


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, clean_targets in loader:
        inputs = inputs.to(device)
        clean_targets = clean_targets.to(device)

        # Calculate ground truth noise residual: Noise = Input - Clean
        noise_target = inputs - clean_targets

        optimizer.zero_grad()

        # Model predicts noise
        noise_pred = model(inputs)

        loss = criterion(noise_pred, noise_target)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Run validation and calculate RMSE.
    """
    model.eval()
    mse_sum = 0.0
    total_pixels = 0

    with torch.no_grad():
        for inputs, clean_targets in loader:
            inputs = inputs.to(device)
            clean_targets = clean_targets.to(device)

            # Predict noise
            noise_pred = model(inputs)

            # Reconstruct clean image: Clean = Input - Noise
            clean_pred = inputs - noise_pred

            # Clip values to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Calculate squared error for RMSE
            # Sum over all pixels in the batch
            batch_mse = torch.sum((clean_pred - clean_targets) ** 2)
            mse_sum += batch_mse.item()
            total_pixels += inputs.numel()

    # Calculate global RMSE
    rmse = np.sqrt(mse_sum / total_pixels)
    return rmse


def run_curriculum_training():
    """
    Execute the two-stage curriculum training strategy.
    """
    start_time = time.time()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Initialize Model
    model = CAResDnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_BLOCKS,
    ).to(device)

    # Loss Function
    criterion = nn.MSELoss()

    # --- Prepare Validation Data (Fixed for both stages) ---
    print("\n[Data] Preparing Validation Set...")
    val_patches, val_targets = extract_patches(
        metadata_path=Config.VAL_METADATA_PATH,
        stride=Config.VAL_STRIDE,
        patch_size=Config.PATCH_SIZE,
        cache_patches_path=Config.CACHE_FILE_VAL,
        cache_targets_path=Config.CACHE_TARGETS_VAL,
        load_cached_data=True,
        is_test=False,
    )
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Helper for Training Loop ---
    def run_stage(stage_name, train_loader, max_epochs, start_epoch=0):
        print(f"\n=== Starting {stage_name} ===")

        # Optimizer and Scheduler
        # Re-initialize optimizer for each stage to adapt to new dataset size/steps
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=Config.ETA_MIN
        )

        best_rmse = float("inf")
        patience_counter = 0

        # If starting stage 2, load best from stage 1 to ensure we start from peak performance
        if stage_name == "Stage 2" and os.path.exists(Config.CHECKPOINT_PATH):
            print(f"Loading best model from previous stage: {Config.CHECKPOINT_PATH}")
            model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=device)
            )
            # Validate immediately to set baseline
            best_rmse = validate(model, val_loader, device)
            print(f"Baseline RMSE: {best_rmse}")

        for epoch in range(start_epoch, start_epoch + max_epochs):
            epoch_start = time.time()

            # Check Time Limit
            elapsed = time.time() - start_time
            if elapsed > Config.MAX_RUNTIME_SECONDS:
                print(
                    f"Time limit reached ({elapsed/3600:.2f} hours). Stopping training."
                )
                break

            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_rmse = validate(model, val_loader, device)

            # Step Scheduler
            scheduler.step()

            epoch_duration = time.time() - epoch_start
            print(
                f"Epoch {epoch+1}/{start_epoch + max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val RMSE: {val_rmse} | "
                f"Time: {epoch_duration:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"New best model saved! RMSE: {best_rmse}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        return best_rmse

    # --- Stage 1: Convergence (Low Density) ---
    print("\n[Data] Preparing Stage 1 Training Data (Stride 20)...")
    train_patches_s1, train_targets_s1 = extract_patches(
        metadata_path=Config.TRAIN_METADATA_PATH,
        stride=Config.STRIDE_STAGE_1,
        patch_size=Config.PATCH_SIZE,
        cache_patches_path=Config.CACHE_FILE_STAGE_1,
        cache_targets_path=Config.CACHE_TARGETS_STAGE_1,
        load_cached_data=True,
        is_test=False,
    )
    train_dataset_s1 = DenoisingDataset(
        train_patches_s1, train_targets_s1, augment=True
    )
    train_loader_s1 = DataLoader(
        train_dataset_s1,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    run_stage("Stage 1", train_loader_s1, Config.MAX_EPOCHS_STAGE_1)

    # --- Stage 2: Refinement (High Density) ---
    # Only proceed if time permits
    if (time.time() - start_time) < Config.MAX_RUNTIME_SECONDS:
        print("\n[Data] Preparing Stage 2 Training Data (Stride 10)...")
        # Free memory from Stage 1
        del train_patches_s1, train_targets_s1, train_dataset_s1, train_loader_s1
        import gc

        gc.collect()

        train_patches_s2, train_targets_s2 = extract_patches(
            metadata_path=Config.TRAIN_METADATA_PATH,
            stride=Config.STRIDE_STAGE_2,
            patch_size=Config.PATCH_SIZE,
            cache_patches_path=Config.CACHE_FILE_STAGE_2,
            cache_targets_path=Config.CACHE_TARGETS_STAGE_2,
            load_cached_data=True,
            is_test=False,
        )
        train_dataset_s2 = DenoisingDataset(
            train_patches_s2, train_targets_s2, augment=True
        )
        train_loader_s2 = DataLoader(
            train_dataset_s2,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        run_stage("Stage 2", train_loader_s2, Config.MAX_EPOCHS_STAGE_2)
    else:
        print("Skipping Stage 2 due to time constraints.")

    print("\nTraining Complete.")
