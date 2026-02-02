import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from library import config, utils, network, data_loader


def train_model(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    load_cached_data=True,
    max_samples=None,
):
    """
    Executes the training pipeline for the denoising model.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for dataloaders.
        learning_rate (float): Initial learning rate.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_samples (int, optional): Maximum number of samples to use (for debugging).
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Initializing training on {device}...")
    print(
        f"Hyperparameters: Epochs={num_epochs}, Batch={batch_size}, LR={learning_rate}"
    )

    # 2. Data Preparation
    # Uses data_loader.prepare_data which handles caching and max_samples logic
    train_patches, train_targets, val_patches, val_targets = data_loader.prepare_data(
        load_cached_data=load_cached_data, max_samples=max_samples
    )

    # 3. Datasets and Loaders
    # Apply augmentation only to training set
    train_dataset = data_loader.DenoisingDataset(
        train_patches, train_targets, augment=config.USE_AUGMENTATION
    )
    val_dataset = data_loader.DenoisingDataset(val_patches, val_targets, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # 4. Model Initialization
    model = network.SE_ZI_ResDnCNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_blocks=config.NUM_BLOCKS,
        kernel_size=config.KERNEL_SIZE,
        padding=config.PADDING,
        use_se=config.USE_SE,
        se_reduction=config.SE_REDUCTION,
        zero_init_residual=config.ZERO_INIT_RESIDUAL,
    ).to(device)

    # 5. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=config.ETA_MIN)

    # Loss function: MSE between predicted noise and actual noise
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training loop...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass (predicts noise residual)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            optimizer.step()

            train_loss_accum += loss.item() * inputs.size(0)

        avg_train_loss = train_loss_accum / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_accum += loss.item() * inputs.size(0)

        avg_val_loss = val_loss_accum / len(val_dataset)

        # Update Scheduler
        scheduler.step()

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_metric": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
