import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import copy

from library.config import Config
from library.utils import seed_everything, dice_coef
from library.dataset import ContrailDataset
from library.model import ContrailUNet
from library.losses import DiceBCELoss


class CheckpointManager:
    """
    Manages saving the top K checkpoints based on a metric (higher is better).
    """

    def __init__(self, checkpoint_dir, top_k=5):
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        # List of tuples: (score, file_path)
        self.checkpoints = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, epoch, score):
        """
        Saves checkpoint if score is in top K.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{score:.6f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # If we haven't filled top_k yet, just save
        if len(self.checkpoints) < self.top_k:
            torch.save(model.state_dict(), filepath)
            self.checkpoints.append((score, filepath))
            self.checkpoints.sort(key=lambda x: x[0])  # Sort by score ascending
            print(f"Saved checkpoint: {filename} (Score: {score:.6f})")
            return

        # If we have top_k, check if current score is better than the worst
        worst_score, worst_path = self.checkpoints[0]
        if score > worst_score:
            # Remove worst file
            if os.path.exists(worst_path):
                os.remove(worst_path)
            self.checkpoints.pop(0)

            # Save new model
            torch.save(model.state_dict(), filepath)
            self.checkpoints.append((score, filepath))
            self.checkpoints.sort(key=lambda x: x[0])
            print(
                f"Saved checkpoint: {filename} (Score: {score:.6f}). Removed: {os.path.basename(worst_path)}"
            )
        else:
            # Not good enough
            pass

    def get_best_checkpoints(self):
        return [path for _, path in self.checkpoints]


def average_checkpoints(checkpoint_paths, output_path):
    """
    Loads state dicts from multiple checkpoints, averages them, and saves the result.
    """
    if not checkpoint_paths:
        print("No checkpoints to average.")
        return

    print(f"Averaging weights from {len(checkpoint_paths)} checkpoints...")

    avg_state_dict = {}

    # Load first checkpoint to initialize
    first_state = torch.load(checkpoint_paths[0], map_location="cpu")
    keys = first_state.keys()

    for key in keys:
        avg_state_dict[key] = first_state[key].clone().float()

    # Add others
    for path in checkpoint_paths[1:]:
        state = torch.load(path, map_location="cpu")
        for key in keys:
            avg_state_dict[key] += state[key].float()

    # Divide
    for key in keys:
        avg_state_dict[key] /= len(checkpoint_paths)

    torch.save(avg_state_dict, output_path)
    print(f"Averaged model saved to {output_path}")


def train_fn(model, loader, optimizer, loss_fn, scaler, device, epoch):
    model.train()

    running_loss = 0.0
    dataset_size = 0

    # Gradient accumulation steps
    accum_iter = Config.ACCUM_ITER

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        with autocast():
            logits = model(images)
            loss = loss_fn(logits, masks)
            # Normalize loss for gradient accumulation
            loss = loss / accum_iter

        scaler.scale(loss).backward()

        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track metrics (multiply by accum_iter to get back original scale for logging)
        running_loss += loss.item() * accum_iter * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_fn(model, loader, device):
    model.eval()

    # Global Dice accumulators
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > 0.5).float()
            targets = masks.float()

            # Flatten for global calculation
            preds = preds.view(-1)
            targets = targets.view(-1)

            intersection = (preds * targets).sum().item()
            union = preds.sum().item() + targets.sum().item()

            total_intersection += intersection
            total_union += union

    # Calculate Global Dice
    smooth = 1e-6
    global_dice = (2.0 * total_intersection) / (total_union + smooth)

    return global_dice


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    train_dataset = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, split="train"
    )
    valid_dataset = ContrailDataset(
        metadata_path=Config.VALIDATION_METADATA_PATH, split="validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")

    # 3. Model
    model = ContrailUNet()
    model.to(device)

    # 4. Optimizer & Scheduler & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    loss_fn = DiceBCELoss(bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT)

    scaler = GradScaler()

    # Checkpoint Manager
    ckpt_manager = CheckpointManager(
        Config.CHECKPOINT_DIR, top_k=Config.TOP_K_CHECKPOINTS
    )

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch
        )

        # Validate
        val_dice = valid_fn(model, valid_loader, device)

        # Step Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Print metrics
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Global Dice: {val_dice:.10f}"
        )

        # Save Checkpoint
        ckpt_manager.save(model, epoch, val_dice)

    # 6. Average Top-K Models
    best_checkpoints = ckpt_manager.get_best_checkpoints()
    if best_checkpoints:
        avg_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        average_checkpoints(best_checkpoints, avg_model_path)

    print("Training complete.")


if __name__ == "__main__":
    # This block is here for local testing if needed, but the prompt says
    # "Only implement the module class/functions. DO NOT include an if __name__ == '__main__': block."
    # However, standard python modules often include this.
    # Based on strict instructions "DO NOT include an if __name__ == '__main__': block",
    # I will comment it out or remove it. The prompt asks for the module content.
    # But usually a script named train.py is executed.
    # I will provide the run_training function which can be called externally.
    pass
