import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    WORKING_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    T_MAX,
    BATCH_SIZE,
    ENSEMBLE_SIZE,
    STREAM_A_CONFIG,
    STREAM_B_CONFIG,
    get_config,
)
from library.models import get_context_specialist, get_texture_specialist
from library.dataset import get_dataloaders
from library.utils import set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Inputs and targets are already inverted and normalized in dataset
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

    return running_loss / count


def train_instance(stream_type, instance_id, debug=False):
    """
    Trains a single instance of the specified stream type.

    Args:
        stream_type (str): 'context' or 'texture'.
        instance_id (int): ID of the model instance (0 to ENSEMBLE_SIZE-1).
        debug (bool): If True, runs for fewer epochs/data for testing.
    """
    # 1. Configuration Setup
    cfg = get_config(debug=debug)

    # Unique seed for this instance to ensure ensemble diversity
    # Base seed 42 + instance_id + offset for stream type
    # Context offset 0, Texture offset 100
    seed_offset = 0 if stream_type == "context" else 100
    current_seed = 42 + instance_id + seed_offset
    set_seed(current_seed)

    print(
        f"\n[{stream_type.upper()} | ID {instance_id}] Starting training with seed {current_seed}..."
    )

    # 2. Model & Data Setup
    if stream_type == "context":
        model = get_context_specialist()
        patch_size = cfg["stream_a"]["patch_size"]
        model_filename = f"context_model_{instance_id}.pth"
    elif stream_type == "texture":
        model = get_texture_specialist()
        patch_size = cfg["stream_b"]["patch_size"]
        model_filename = f"texture_model_{instance_id}.pth"
    else:
        raise ValueError(f"Unknown stream_type: {stream_type}")

    model = model.to(DEVICE)

    # Get Dataloaders
    # Note: Validation loader is always full-image, no patching
    train_loader, val_loader = get_dataloaders(
        batch_size=cfg["batch_size"],
        patch_size=patch_size,
        mode="train",
        load_cached_data=True,
    )

    # 3. Optimization Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Cosine Annealing Scheduler
    # T_max corresponds to the total number of epochs to decay LR to 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["t_max"])

    # 4. Training Loop
    total_epochs = cfg["epochs"]

    # If debug, reduce epochs
    if debug:
        total_epochs = 2
        print("Debug mode: Reducing epochs to 2.")

    start_time = time.time()

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Calculate RMSE for reporting
        train_rmse = math.sqrt(train_loss)
        val_rmse = math.sqrt(val_loss)

        # Print metrics (every 50 epochs or first/last)
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == total_epochs - 1:
            print(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train MSE: {train_loss:.6f} (RMSE: {train_rmse:.6f}) | "
                f"Val MSE: {val_loss:.6f} (RMSE: {val_rmse:.6f})"
            )

    total_time = time.time() - start_time
    print(
        f"[{stream_type.upper()} | ID {instance_id}] Training complete in {total_time:.2f}s."
    )

    # 5. Save Model
    # We prioritize fully converged models, so we save the final state.
    save_path = os.path.join(WORKING_DIR, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_ensemble(debug=False):
    """
    Orchestrates the training of the entire ensemble.
    Trains ENSEMBLE_SIZE instances of Context Specialist
    and ENSEMBLE_SIZE instances of Texture Specialist.
    """
    cfg = get_config(debug=debug)
    ensemble_size = cfg["ensemble_size"]

    if debug:
        ensemble_size = 1
        print("Debug mode: Reducing ensemble size to 1 per stream.")

    print("=" * 60)
    print(f"Starting Ensemble Training: {ensemble_size * 2} models total.")
    print("=" * 60)

    # Train Stream A: Context Specialists
    for i in range(ensemble_size):
        train_instance("context", i, debug=debug)

    # Train Stream B: Texture Specialists
    for i in range(ensemble_size):
        train_instance("texture", i, debug=debug)

    print("\nAll models trained successfully.")


def train_model(debug=False):
    """
    Main entry point for the training module.
    """
    train_ensemble(debug=debug)
