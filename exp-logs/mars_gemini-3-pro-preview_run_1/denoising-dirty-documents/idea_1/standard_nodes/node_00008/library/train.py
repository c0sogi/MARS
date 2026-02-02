import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import UNet, train_one_epoch, validate, generate_submission


def train_model(epochs=None, batch_size=None, dataset_limit=None):
    """
    Manages the training workflow, including model initialization, training loop,
    validation, checkpointing, early stopping, and submission generation.

    Args:
        epochs (int, optional): Number of training epochs. Defaults to Config.NUM_EPOCHS.
        batch_size (int, optional): Batch size for training. Defaults to Config.BATCH_SIZE.
        dataset_limit (int, optional): Limit the number of samples for debugging.
    """
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Handle Hyperparameter Overrides
    if epochs is None:
        epochs = Config.NUM_EPOCHS

    # Update Config.BATCH_SIZE if provided, as get_dataloaders reads from Config
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    # 3. Data Preparation
    # Note: No caching needed here as data is loaded/processed on-the-fly by the Dataset class
    train_loader, val_loader, test_loader = get_dataloaders(dataset_limit=dataset_limit)

    # 4. Model Initialization
    model = UNet(n_channels=Config.NUM_CHANNELS, n_classes=1, bilinear=True)
    model.to(device)

    # 5. Optimization Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # Cite solution_lesson_node_00006: Cosine Annealing scheduler
    # Cite solution_lesson_node_00007: Decoupled horizon
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=1e-6
    )

    # 6. Training Loop
    best_val_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Step the scheduler
        scheduler.step()

        # Log metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss (MSE): {train_loss} - Val RMSE: {val_rmse}"
        )

        # Checkpointing
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 7. Inference and Submission
    print(f"Loading best model (RMSE: {best_val_rmse}) for inference...")

    # Load best weights
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    # Generate submission file
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
