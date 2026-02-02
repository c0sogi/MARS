import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, rmse_loss, pad_to_multiple, unpad
from library.dataset import get_dataloaders
from library.model import UNet


def run_training(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Executes the training pipeline for the Denoising Task.
    Trains an ensemble of models based on Config.SEEDS.

    Args:
        epochs (int): Number of training epochs.
        debug (bool): Whether to run in debug mode (fewer samples).
    """
    # Update Configuration based on arguments
    Config.DEBUG = debug
    Config.EPOCHS = epochs
    Config.T_MAX = epochs  # Ensure scheduler aligns with the new epoch count

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load DataLoaders
    # The get_dataloaders function handles caching internally based on Config settings
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    device = Config.DEVICE

    # Iterate over seeds for the ensemble
    for seed in Config.SEEDS:
        print(f"Starting training for seed: {seed}")
        seed_everything(seed)

        # Initialize Model
        model = UNet(n_channels=1, n_classes=1).to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        # T_max matches the total epochs for proper cosine decay to zero
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Loss Function
        # Using MSE Loss for optimization (equivalent to minimizing RMSE)
        criterion = nn.MSELoss()

        # Training Loop
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch in train_loader:
                # Unpack batch: noisy_image, clean_image, image_id
                noisy = batch[0].to(device)
                clean = batch[1].to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(noisy)

                # Compute loss
                loss = criterion(outputs, clean)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_steps += 1

            # Step the scheduler after each epoch
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Validation Loop
            model.eval()
            val_rmse_sum = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    noisy = batch[0].to(device)
                    clean = batch[1].to(device)

                    # Pad input to be divisible by 2^depth
                    divisor = 2**Config.MODEL_DEPTH
                    padded_noisy, padding_info = pad_to_multiple(noisy, divisor=divisor)

                    # Inference
                    outputs = model(padded_noisy)

                    # Remove padding to match original dimensions for metric calculation
                    outputs = unpad(outputs, padding_info)

                    # Calculate RMSE for this image
                    val_rmse = rmse_loss(outputs, clean)
                    val_rmse_sum += val_rmse.item()
                    val_steps += 1

            # Calculate average metrics
            avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
            avg_val_rmse = val_rmse_sum / val_steps if val_steps > 0 else 0.0

            # Print metrics with full precision
            print(
                f"Seed {seed} | Epoch {epoch + 1} | LR: {current_lr} | Train MSE: {avg_train_loss} | Val RMSE: {avg_val_rmse}"
            )

        # Save the fully converged model state
        save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
