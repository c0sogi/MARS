import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, calculate_iou_map
from library.dataset import SaltDataset, get_transforms
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss


class SaltTrainer:
    """
    Trainer class for Salt Segmentation Task using High-Fidelity Training Regime.
    Handles AMP, Deep Supervision, Loss Switching, and Threshold Optimization.
    """

    def __init__(self):
        self.config = Config
        self.device = torch.device(self.config.DEVICE)

        # Ensure reproducibility
        seed_everything(self.config.SEED)

        # Create directories
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)

        # Initialize Model
        print(
            f"Initializing model: {self.config.MODEL_NAME} with {self.config.ENCODER}..."
        )
        self.model = SaltUNetPlusPlus().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
            min_lr=self.config.MIN_LR,
        )

        # Loss Functions
        self.criterion_bce_dice = BCEDiceLoss().to(self.device)
        self.criterion_lovasz = LovaszHingeLoss().to(self.device)

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=self.config.AMP)

        # Data Loaders
        self._init_dataloaders()

        # State tracking
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.early_stopping_counter = 0

    def _init_dataloaders(self):
        print("Initializing dataloaders...")

        train_dataset = SaltDataset(
            mode="train", transform=get_transforms(mode="train"), load_cached_data=True
        )

        val_dataset = SaltDataset(
            mode="val", transform=get_transforms(mode="val"), load_cached_data=True
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        # Determine which loss function to use
        # Before LOVASZ_EPOCH: BCE + Dice (Warmup)
        # After LOVASZ_EPOCH: Lovasz Hinge (Fine-tuning)
        if epoch < self.config.LOVASZ_EPOCH:
            criterion = self.criterion_bce_dice
            loss_name = "BCE+Dice"
        else:
            criterion = self.criterion_lovasz
            loss_name = "Lovasz"

        start_time = time.time()

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=self.config.AMP):
                outputs = self.model(images)
                loss = criterion(outputs, masks)

            # Backward Pass
            self.scaler.scale(loss).backward()

            # Gradient Accumulation (if configured > 1, currently 1)
            if (batch_idx + 1) % self.config.GRAD_ACCUMULATION == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{self.config.EPOCHS} [{loss_name}] | "
            f"Train Loss: {epoch_loss:.8f} | Time: {duration:.2f}s"
        )

        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        # Containers for full dataset evaluation
        all_preds = []
        all_masks = []

        # Use BCE+Dice for validation loss tracking regardless of epoch
        # to maintain a consistent loss metric, though we optimize for mAP
        criterion = self.criterion_bce_dice

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # In eval mode, model returns single tensor (final output)
                # We need to unsqueeze it if loss expects list, or handle it directly.
                # BCEDiceLoss handles single tensor fine.
                with autocast(enabled=self.config.AMP):
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)

                running_loss += loss.item()

                # Apply sigmoid to get probabilities for metric calculation
                preds = torch.sigmoid(outputs)

                # Store predictions and targets on CPU to save GPU memory
                all_preds.append(preds.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Remove channel dim: (B, 1, H, W) -> (B, H, W)
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_masks.ndim == 4:
            all_masks = all_masks.squeeze(1)

        # Optimize Threshold
        # Sweep over probability thresholds to find the one that maximizes mAP
        thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
        best_threshold_score = -1.0
        best_threshold = 0.5

        # We can calculate this on a subset if speed is an issue, but for 600 val images it's fast.
        for t in thresholds:
            score = calculate_iou_map(all_preds, all_masks, pixel_threshold=t)
            if score > best_threshold_score:
                best_threshold_score = score
                best_threshold = t

        print(
            f"Validation | Loss: {val_loss:.8f} | "
            f"Max mAP: {best_threshold_score:.8f} at threshold {best_threshold:.2f}"
        )

        return val_loss, best_threshold_score, best_threshold

    def fit(self):
        print(f"Starting training for {self.config.EPOCHS} epochs...")

        for epoch in range(self.config.EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_score, best_thresh = self.validate()

            # Scheduler Step
            # Reduce LR if validation score plateaus
            self.scheduler.step(val_score)

            # Save Best Model
            if val_score > self.best_score:
                print(
                    f"Score Improved: {self.best_score:.8f} -> {val_score:.8f}. Saving model..."
                )
                self.best_score = val_score
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(
                    f"No improvement. EarlyStopping counter: {self.early_stopping_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            # Early Stopping
            # Ensure we don't stop before the Lovasz fine-tuning phase has had a chance
            if (
                self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE
            ) and (epoch > self.config.LOVASZ_EPOCH + 5):
                print("Early stopping triggered.")
                break

            print("-" * 50)

        print(
            f"Training complete. Best Score: {self.best_score:.8f} at Epoch {self.best_epoch+1}"
        )
