import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import GlobalConfig
from library.utils import seed_everything
from library.model import UNet
from library.dataset import get_dataloader


def train_model(stream_config, seed, epochs=GlobalConfig.EPOCHS, debug=False):
    """
    Trains a single model instance (Stream A or Stream B) with a specific seed.

    Args:
        stream_config: Configuration class (StreamAConfig or StreamBConfig).
        seed (int): Random seed for initialization and data shuffling.
        epochs (int): Number of training epochs.
        debug (bool): If True, runs a shortened loop for debugging.
    """

    # 1. Setup Reproducibility
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Starting training for {stream_config.NAME} | Seed: {seed} | Device: {device}"
    )

    # 2. Prepare DataLoaders
    # Train loader needs stream_config for specific patch sizes
    train_loader = get_dataloader(
        mode="train",
        stream_config=stream_config,
        batch_size=GlobalConfig.BATCH_SIZE,
        shuffle=True,
    )

    # Val loader (batch_size=1 handled internally for variable sizes)
    val_loader = get_dataloader(
        mode="val", stream_config=None, shuffle=False  # Not needed for val
    )

    # 3. Initialize Model
    model = UNet(
        depth=stream_config.DEPTH,
        encoder_filters=stream_config.ENCODER_FILTERS,
        bottleneck_filters=stream_config.BOTTLENECK_FILTERS,
        bottleneck_depth=stream_config.BOTTLENECK_DEPTH,
        in_channels=1,
        out_channels=1,
    ).to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=GlobalConfig.LEARNING_RATE)

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    criterion = nn.MSELoss()

    # 5. Training Loop
    best_rmse = float("inf")
    save_path = os.path.join(
        GlobalConfig.WORKING_DIR, f"{stream_config.NAME}_seed_{seed}.pth"
    )

    # Ensure working directory exists
    os.makedirs(GlobalConfig.WORKING_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for i, (noisy, clean, _) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            outputs = model(noisy)
            loss = criterion(outputs, clean)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

            if debug and i >= 5:
                break

        avg_train_loss = train_loss_accum / train_batches if train_batches > 0 else 0.0

        # 6. Validation Loop
        model.eval()
        total_squared_error = 0.0
        total_pixels = 0

        with torch.no_grad():
            for i, (noisy, clean, _) in enumerate(val_loader):
                noisy = noisy.to(device)
                clean = clean.to(device)

                outputs = model(noisy)

                # Calculate squared error sum for this image
                # We use reduction='sum' to aggregate errors, then divide by total pixels later for global RMSE
                batch_mse_sum = nn.functional.mse_loss(outputs, clean, reduction="sum")

                total_squared_error += batch_mse_sum.item()
                total_pixels += clean.numel()

                if debug and i >= 5:
                    break

        # Calculate Global RMSE
        if total_pixels > 0:
            global_mse = total_squared_error / total_pixels
            val_rmse = np.sqrt(global_mse)
        else:
            val_rmse = float("inf")

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr} | Train Loss: {avg_train_loss} | Val RMSE: {val_rmse}"
        )

        # Save Best Model
        if val_rmse < best_rmse:
            print(
                f"Validation RMSE improved from {best_rmse} to {val_rmse}. Saving model to {save_path}"
            )
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)

        if debug:
            print("Debug mode active: stopping after 1 epoch.")
            break

    print(f"Training complete. Best Val RMSE: {best_rmse}")
    return best_rmse
