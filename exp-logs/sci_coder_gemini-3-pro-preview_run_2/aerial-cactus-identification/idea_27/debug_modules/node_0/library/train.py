import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.data import get_loaders
from library.model import CactusNet, train_one_epoch, validate, generate_submission


def fit_model(seed, train_loader, val_loader, device, epochs=Config.NUM_EPOCHS):
    """
    Manages the training loop for a specific seed.
    Initializes model, optimizer, scheduler, and saves the best checkpoint.
    """
    print(f"\n--- Training Seed {seed} ---")
    seed_everything(seed)

    # Initialize Model
    model = CactusNet(num_classes=Config.NUM_CLASSES).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(epochs):
        # Train and Validate using imported functions
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
        )

        # Save Best Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": best_auc,
                },
                f"model_seed_{seed}.pth",
            )

    print(f"Best AUC for Seed {seed}: {best_auc:.6f}")


def main():
    """
    Main execution routine.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load Data
    # Caching is handled internally by get_loaders in library.data
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Train models for each seed
    for seed in Config.SEEDS:
        fit_model(seed, train_loader, val_loader, device)

    # Generate Submission
    # Uses the checkpoints saved by fit_model
    generate_submission(test_loader, device)


if __name__ == "__main__":
    main()
