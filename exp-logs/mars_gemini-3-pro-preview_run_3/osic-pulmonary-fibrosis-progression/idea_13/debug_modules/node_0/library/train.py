import os
import torch
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DSPRNet, train_one_epoch, validate


def run_training(
    epochs=Config.EPOCHS, load_cached_data=True, debug=False, patience=Config.PATIENCE
):
    """
    Executes the training pipeline for the DSPR-Net model.

    Args:
        epochs (int): Total number of training epochs.
        load_cached_data (bool): Whether to use cached preprocessed images.
        debug (bool): If True, limits the dataset size for rapid debugging.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        str: Path to the best saved model checkpoint.
    """
    # 1. Setup Reproducibility and Device
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Configure Debugging
    # Modifying Config directly affects get_dataloaders behavior
    if debug:
        print("DEBUG MODE: Restricting training to 100 samples.")
        Config.MAX_TRAIN_SAMPLES = 100
    else:
        Config.MAX_TRAIN_SAMPLES = None

    # 3. Load Data
    # target_scaler is needed for inverse transforming predictions during validation
    train_loader, val_loader, target_scaler, _ = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 4. Initialize Model
    model = DSPRNet().to(device)

    # 5. Optimizer Setup with Differential Learning Rates
    # We separate the pre-trained backbone from the newly initialized heads
    backbone_params = list(model.img_branch.backbone.parameters())

    # Gather parameters for the projection, dual streams, and final head
    head_params = (
        list(model.img_branch.projection.parameters())
        + list(model.stream_a.parameters())
        + list(model.stream_b.parameters())
        + list(model.head.parameters())
    )

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 6. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # 7. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Perform one epoch of training
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Evaluate on validation set
        val_score = validate(model, val_loader, device, target_scaler)

        # Update learning rate
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Checkpoint and Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (Patience: {patience})")
            break

    print(f"Training complete. Best Val Score: {best_score}")
    return best_model_path
