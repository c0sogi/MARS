import os
import time
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import library.config as config
import library.utils as utils
from library.data_interface import DataInterface
from library.dataset import BEVDataset
from library.model import BEVDetector
from library.loss import BEVLoss


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, device, epoch_idx
):
    """
    Executes one training epoch.
    """
    model.train()
    running_stats = {}
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        inputs = batch["input"].to(device)

        # Targets
        targets = {
            "hm": batch["hm"].to(device),
            "reg": batch["reg"].to(device),
            "reg_mask": batch["reg_mask"].to(device),
        }

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Loss calculation
        loss, stats = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate stats
        for k, v in stats.items():
            if k not in running_stats:
                running_stats[k] = 0.0
            running_stats[k] += v

    # Average stats
    avg_stats = {k: v / num_batches for k, v in running_stats.items()}

    epoch_time = time.time() - start_time
    print(
        f"Epoch [{epoch_idx+1}] Train Time: {epoch_time:.2f}s | "
        f"Loss: {avg_stats['loss']:.10f} | "
        f"HM: {avg_stats['hm_loss']:.10f} | "
        f"WH: {avg_stats['wh_loss']:.10f} | "
        f"Off: {avg_stats['off_loss']:.10f} | "
        f"Z: {avg_stats['z_loss']:.10f} | "
        f"Rot: {avg_stats['rot_loss']:.10f}"
    )

    return avg_stats


def validate_epoch(model, dataloader, criterion, device, epoch_idx):
    """
    Executes one validation epoch.
    """
    model.eval()
    running_stats = {}
    num_batches = len(dataloader)

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["input"].to(device)

            targets = {
                "hm": batch["hm"].to(device),
                "reg": batch["reg"].to(device),
                "reg_mask": batch["reg_mask"].to(device),
            }

            preds = model(inputs)
            loss, stats = criterion(preds, targets)

            for k, v in stats.items():
                if k not in running_stats:
                    running_stats[k] = 0.0
                running_stats[k] += v

    avg_stats = {k: v / num_batches for k, v in running_stats.items()}

    epoch_time = time.time() - start_time
    print(
        f"Epoch [{epoch_idx+1}] Val Time: {epoch_time:.2f}s   | "
        f"Loss: {avg_stats['loss']:.10f} | "
        f"HM: {avg_stats['hm_loss']:.10f} | "
        f"WH: {avg_stats['wh_loss']:.10f} | "
        f"Off: {avg_stats['off_loss']:.10f} | "
        f"Z: {avg_stats['z_loss']:.10f} | "
        f"Rot: {avg_stats['rot_loss']:.10f}"
    )

    return avg_stats


def train_model(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    load_cached_data=True,
    num_workers=config.NUM_WORKERS,
):
    """
    Main function to train the BEV Object Detector.
    """
    logger = utils.get_logger(__name__)
    logger.info("Starting training pipeline...")

    # 1. Setup
    config.set_seed(config.SEED)
    device = config.get_device()
    logger.info(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 2. Data Interface
    logger.info("Initializing Data Interface...")
    data_interface = DataInterface(load_cached_data=load_cached_data)

    # 3. Datasets & Dataloaders
    logger.info("Preparing Datasets...")
    train_dataset = BEVDataset(
        split="train", data_interface=data_interface, load_cached_data=load_cached_data
    )
    val_dataset = BEVDataset(
        split="val", data_interface=data_interface, load_cached_data=load_cached_data
    )

    # Reproducibility for workers (Cite solution_lesson_node_00010)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Model, Loss, Optimizer
    logger.info("Initializing Model...")
    model = BEVDetector(
        num_classes=config.NUM_CLASSES,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
    )
    model = model.to(device)

    criterion = BEVLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    logger.info("Starting Training Loop...")

    for epoch in range(num_epochs):
        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )

        # Validate
        val_stats = validate_epoch(model, val_loader, criterion, device, epoch)

        current_val_loss = val_stats["loss"]

        # Checkpointing
        if current_val_loss < best_val_loss - config.MIN_DELTA:
            logger.info(
                f"Validation loss improved from {best_val_loss:.10f} to {current_val_loss:.10f}. Saving model..."
            )
            best_val_loss = current_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{config.PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")

    # Load best model before returning
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
