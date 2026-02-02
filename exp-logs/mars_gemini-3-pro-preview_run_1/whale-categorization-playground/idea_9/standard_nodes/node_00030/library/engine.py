import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, AverageMeter, save_checkpoint, calculate_map5
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet
from library.loss import LabelSmoothingCrossEntropy


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()

    # Iterate over the dataloader
    # Note: Progress bars are suppressed as per requirements
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        # For ArcFace training, we pass the targets to the model
        outputs = model(images, labels=targets)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_with_tta(val_loader, model, device):
    """
    Validates the model using Test-Time Augmentation (Horizontal Flip).
    Returns the Mean Average Precision @ 5.
    """
    model.eval()

    total_map5_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # 1. Forward pass with original images
            # Pass labels=None to get inference logits (cosine similarities)
            logits_orig = model(images, labels=None)

            # 2. Forward pass with horizontally flipped images
            # Dimension 3 is width in (N, C, H, W)
            images_flip = torch.flip(images, [3])
            logits_flip = model(images_flip, labels=None)

            # 3. Average predictions
            avg_logits = (logits_orig + logits_flip) / 2.0

            # 4. Calculate MAP@5 for this batch
            # calculate_map5 returns the average score for the batch
            batch_score = calculate_map5(avg_logits, targets)

            total_map5_sum += batch_score * batch_size
            total_samples += batch_size

    final_map5 = total_map5_sum / total_samples if total_samples > 0 else 0.0
    return final_map5


def train_model(seed, debug=Config.DEBUG):
    """
    Main training routine for a single model instance.

    Args:
        seed (int): Random seed for this training run.
        debug (bool): Whether to run in debug mode (fewer samples).
    """
    # 1. Reproducibility
    seed_everything(seed)

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    train_dataset = WhaleDataset(
        mode="train", transform=get_transforms("train"), debug=debug
    )
    val_dataset = WhaleDataset(mode="val", transform=get_transforms("val"), debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Seed {seed}: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    # 3. Model Initialization
    model = WhaleDenseNet(pretrained=True)
    model.to(device)

    # 4. Optimization
    criterion = LabelSmoothingCrossEntropy(smoothing=Config.LABEL_SMOOTHING)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    best_map5 = 0.0
    patience_counter = 0

    # Create seed-specific output directory
    seed_output_dir = os.path.join(Config.WORKING_DIR, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    print(f"Starting training for Seed {seed}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate (with TTA)
        val_map5 = validate_with_tta(val_loader, model, device)

        # Scheduler step
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAP@5: {val_map5:.10f}"
        )

        # Checkpointing
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_map5": best_map5,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            output_dir=seed_output_dir,
        )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val MAP@5: {best_map5:.10f}"
            )
            break

    print(f"Finished training for Seed {seed}. Best MAP@5: {best_map5:.10f}")
    return best_map5
