import os
import gc
import time
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import set_seed
from library.loss import DiceBCELoss
from library.data import get_loaders
from library.model import ContrailUnet


class Trainer:
    """
    Trainer class for the Contrail Segmentation model.
    Handles training loop, validation, checkpointing, and metric tracking.
    """

    def __init__(self, debug=False):
        """
        Args:
            debug (bool): If True, runs on a subset of data for debugging.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # --- Data Loaders ---
        loaders = get_loaders(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=debug
        )
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        # --- Model ---
        self.model = ContrailUnet()
        self.model.to(self.device)

        # --- Loss Function ---
        self.criterion = DiceBCELoss()

        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # --- Scheduler ---
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # --- Mixed Precision ---
        self.scaler = GradScaler()

        # --- Checkpointing State ---
        # Min-heap to store (dice, epoch, filepath)
        # We use a min-heap so the 0-th element is always the one with the lowest Dice score
        # among the top K, making it efficient to replace.
        self.top_k_checkpoints = []

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        elapsed = time.time() - start_time

        return epoch_loss, elapsed

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        Computes Global Dice Coefficient.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        # Accumulators for Global Dice Calculation
        intersection_sum = 0.0
        union_sum = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                batch_size = images.size(0)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Calculate Dice Statistics
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > Config.THRESHOLD).float()

                # Accumulate intersection and union for Global Dice
                intersection_sum += (preds_bin * masks).sum().item()
                union_sum += preds_bin.sum().item() + masks.sum().item()

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        # Compute Global Dice
        smooth = 1e-6
        global_dice = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

        return epoch_loss, global_dice

    def save_checkpoint(self, epoch, dice):
        """
        Saves checkpoint and maintains only the Top-K best checkpoints based on Dice score.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{dice:.6f}.pth"
        save_path = os.path.join(Config.CHECKPOINT_DIR, filename)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "dice": dice,
        }

        # Save the current checkpoint first
        torch.save(checkpoint, save_path)

        entry = (dice, epoch, save_path)

        if len(self.top_k_checkpoints) < Config.TOP_K_CHECKPOINTS:
            # If we haven't filled the top K yet, just push
            heapq.heappush(self.top_k_checkpoints, entry)
        else:
            # If the current dice is better than the worst in our top K (the root of min-heap)
            if dice > self.top_k_checkpoints[0][0]:
                # Remove the worst one from heap and get its path
                _, _, path_to_remove = heapq.heapreplace(self.top_k_checkpoints, entry)

                # Delete the physical file of the removed checkpoint
                if os.path.exists(path_to_remove):
                    try:
                        os.remove(path_to_remove)
                    except OSError:
                        pass
            else:
                # Current checkpoint is worse than all top K.
                # Since we already saved it to disk above, we should delete it now to save space.
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except OSError:
                        pass

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            # --- Train ---
            train_loss, train_time = self.train_one_epoch(epoch)

            # --- Validate ---
            val_loss, val_dice = self.validate(epoch)

            # --- Scheduler Step ---
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Logging ---
            # Printing full precision as requested
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val Dice: {val_dice:.10f} | "
                f"LR: {current_lr:.8f} | "
                f"Time: {train_time:.2f}s"
            )

            # --- Checkpointing ---
            self.save_checkpoint(epoch, val_dice)

            # --- Cleanup ---
            gc.collect()
            torch.cuda.empty_cache()
