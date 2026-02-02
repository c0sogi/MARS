import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.dataset import get_dataloaders
from library.models import UNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (average_loss, rmse_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for RMSE calculation
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        pred_tensor = torch.cat(all_preds)
        target_tensor = torch.cat(all_targets)
        rmse = calculate_rmse(target_tensor, pred_tensor)
    else:
        rmse = 0.0

    return epoch_loss, rmse


def run_training_session(stream_config, seed, load_cached_data=True):
    """
    Runs a full training session for a specific stream configuration and seed.

    Implements the "Converged Independent Training" strategy:
    - Runs for full Config.NUM_EPOCHS.
    - Uses Cosine Annealing scheduler.
    - Saves the model with the best validation RMSE.

    Args:
        stream_config (dict): Configuration dictionary for the specific stream (A or B).
        seed (int): Random seed for this specific model instance.
        load_cached_data (bool): Whether to use cached data loading.

    Returns:
        str: Path to the saved best model.
    """
    # 1. Setup
    seed_everything(seed)
    device = torch.device(Config.DEVICE)

    # Define model save path
    model_name = f"{stream_config['name']}_seed_{seed}.pth"
    save_path = os.path.join(Config.WORKING_DIR, model_name)

    print(f"\n[{stream_config['name']}] Starting training for Seed {seed}")
    print(
        f"Configuration: Depth={stream_config['depth']}, Patch={stream_config['patch_size']}"
    )
    print(f"Output Model: {save_path}")

    # 2. Data
    train_loader, val_loader = get_dataloaders(
        stream_config, load_cached_data=load_cached_data
    )

    # 3. Model
    model = UNet(
        n_channels=1,
        n_classes=1,
        depth=stream_config["depth"],
        base_channels=stream_config["base_channels"],
    ).to(device)

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Loss function (MSE as per config)
    if Config.LOSS_FN == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()  # Default fallback

    # Scheduler: Cosine Annealing decoupled from early stopping
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 5. Training Loop
    best_rmse = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Checkpointing: Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            # Print update
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse}"
            )
        elif (epoch + 1) % 50 == 0:
            # Periodic logging for monitoring
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse}"
            )

    print(f"Training finished for {model_name}. Best RMSE: {best_rmse}")
    return save_path


def train_all_models(load_cached_data=True):
    """
    Iterates through all streams and seeds defined in Config to train the full ensemble.
    """
    trained_models = []

    for stream in Config.STREAMS:
        print(f"\n=== Processing Stream: {stream['name']} ===")
        for seed in stream["seeds"]:
            model_path = run_training_session(stream, seed, load_cached_data)
            trained_models.append(model_path)

    print(f"\nAll models trained. Total models: {len(trained_models)}")
    return trained_models
