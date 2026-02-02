import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PillarUNet3D


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()

    total_loss = 0.0
    total_hm_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        batch["voxels"] = batch["voxels"].to(device)
        batch["num_points"] = batch["num_points"].to(device)
        batch["coordinates"] = batch["coordinates"].to(device)

        # Targets
        if "hm" in batch:
            batch["hm"] = batch["hm"].to(device)
            batch["ind"] = batch["ind"].to(device)
            batch["mask"] = batch["mask"].to(device)
            batch["cat"] = batch["cat"].to(device)
            batch["target_reg"] = batch["target_reg"].to(device)

        optimizer.zero_grad()

        # Forward pass (loss is calculated inside model if targets provided)
        _, loss, stats = model(batch)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        # Accumulate stats
        total_loss += stats["total_loss"]
        total_hm_loss += stats["hm_loss"]
        total_reg_loss += stats["reg_loss"]
        num_batches += 1

    avg_stats = {
        "total_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "hm_loss": total_hm_loss / num_batches if num_batches > 0 else 0.0,
        "reg_loss": total_reg_loss / num_batches if num_batches > 0 else 0.0,
    }

    return avg_stats


def validate(model, dataloader, device):
    """
    Runs validation on the provided dataloader.
    """
    model.eval()

    total_loss = 0.0
    total_hm_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch["voxels"] = batch["voxels"].to(device)
            batch["num_points"] = batch["num_points"].to(device)
            batch["coordinates"] = batch["coordinates"].to(device)

            # Targets
            if "hm" in batch:
                batch["hm"] = batch["hm"].to(device)
                batch["ind"] = batch["ind"].to(device)
                batch["mask"] = batch["mask"].to(device)
                batch["cat"] = batch["cat"].to(device)
                batch["target_reg"] = batch["target_reg"].to(device)

            # Forward pass
            _, loss, stats = model(batch)

            # Accumulate stats
            total_loss += stats["total_loss"]
            total_hm_loss += stats["hm_loss"]
            total_reg_loss += stats["reg_loss"]
            num_batches += 1

    avg_stats = {
        "total_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "hm_loss": total_hm_loss / num_batches if num_batches > 0 else 0.0,
        "reg_loss": total_reg_loss / num_batches if num_batches > 0 else 0.0,
    }

    return avg_stats


def train_model(
    num_epochs=Config.NUM_EPOCHS, debug_limit=None, load_cached_data=True, patience=5
):
    """
    Main function to train the model.

    Args:
        num_epochs (int): Number of epochs to train.
        debug_limit (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to use cached intermediate files.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Override Config debug limit if provided
    if debug_limit is not None:
        Config.DEBUG = True
        Config.DEBUG_SAMPLE_SIZE = debug_limit

    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")

    # 2. DataLoaders
    print("Initializing Datasets...")
    train_dataset = NuScenesDataset(is_train=True, load_cached_data=load_cached_data)
    val_dataset = NuScenesDataset(is_train=False, load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Model & Optimizer
    model = PillarUNet3D().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_stats = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Logging
        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_stats['total_loss']}")
        print(f"  Heatmap: {train_stats['hm_loss']}")
        print(f"  Regression: {train_stats['reg_loss']}")
        print(f"Val Loss: {val_stats['total_loss']}")
        print(f"  Heatmap: {val_stats['hm_loss']}")
        print(f"  Regression: {val_stats['reg_loss']}")

        # Checkpointing & Early Stopping
        current_val_loss = val_stats["total_loss"]

        # Save Latest
        latest_path = os.path.join(Config.CHECKPOINT_DIR, "latest_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_val_loss,
            },
            latest_path,
        )

        # Save Best
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                best_path,
            )
            print(f"New best model saved to {best_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss}")
