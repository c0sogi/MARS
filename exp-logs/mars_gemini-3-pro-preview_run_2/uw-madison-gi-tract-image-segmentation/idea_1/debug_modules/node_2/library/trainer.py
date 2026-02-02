import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.dataset import UWDataset
from library.model import MobileNetV2UNet
from library.loss import BCEDiceLoss
from library.utils import set_seed, dice_coef, rle_encode


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the segmentation model.
    """

    def __init__(self, debug=False, load_cached_data=True):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, uses a smaller subset of data for debugging.
            load_cached_data (bool): If True, attempts to load processed metadata from cache.
        """
        self.debug = debug
        self.load_cached_data = load_cached_data
        self.device = Config.DEVICE

        # Ensure working directories exist
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    def get_dataloaders(self):
        """
        Creates and returns training and validation dataloaders.
        """
        train_dataset = UWDataset(
            mode="train", debug=self.debug, load_cached_data=self.load_cached_data
        )
        val_dataset = UWDataset(
            mode="val", debug=self.debug, load_cached_data=self.load_cached_data
        )

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
            drop_last=False,
        )

        return train_loader, val_loader

    def train_one_epoch(self, model, loader, optimizer, criterion, scaler):
        """
        Runs one epoch of training.
        """
        model.train()
        running_loss = 0.0
        dataset_size = len(loader.dataset)

        for images, masks in loader:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            optimizer.zero_grad()

            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.scale(optimizer).step()
            scaler.update()

            # Aggregate loss (multiply by batch size to get total, then divide by dataset size later)
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, model, loader, criterion):
        """
        Runs validation on the validation set.
        Returns average loss and average Dice coefficient.
        """
        model.eval()
        running_loss = 0.0
        running_dice = 0.0
        dataset_size = len(loader.dataset)

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                with autocast(enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

                # Calculate Dice Score
                # Apply Sigmoid because model returns logits
                probs = torch.sigmoid(outputs)
                preds_binary = (probs > Config.PRED_THRESHOLD).float()

                # dice_coef computes score over flattened arrays (Micro-average over batch)
                # We weight it by batch size to compute the global average later
                batch_dice = dice_coef(masks, preds_binary)
                running_dice += batch_dice * images.size(0)

        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        return epoch_loss, epoch_dice

    def fit(self, epochs=None):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        if epochs is None:
            epochs = Config.EPOCHS

        set_seed(Config.SEED)

        # Data
        train_loader, val_loader = self.get_dataloaders()

        # Model
        model = MobileNetV2UNet(pretrained=Config.PRETRAINED)
        model = model.to(self.device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
        )

        # Loss & Scaler
        criterion = BCEDiceLoss()
        scaler = GradScaler()

        # Tracking
        best_dice = 0.0
        patience = 5
        patience_counter = 0

        print(f"Starting training on {self.device} for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, criterion, scaler
            )

            # Validate
            val_loss, val_dice = self.validate(model, val_loader, criterion)

            # Step Scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val Dice: {val_dice:.8f}"
            )

            # Checkpoint
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  >>> New Best Model! Saved to {Config.MODEL_SAVE_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s. Best Val Dice: {best_dice:.8f}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model
        and saves the submission CSV.
        """
        print("Generating submission...")

        # Load Data
        test_dataset = UWDataset(
            mode="test", debug=self.debug, load_cached_data=self.load_cached_data
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Load Model
        model = MobileNetV2UNet(
            pretrained=False
        )  # Pretrained weights not needed for loading state_dict
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        model = model.to(self.device)
        model.eval()

        results = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device, dtype=torch.float32)

                with autocast(enabled=True):
                    outputs = model(images)

                # Apply Sigmoid and Threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > Config.PRED_THRESHOLD).cpu().numpy().astype(np.uint8)

                # Iterate through batch
                for i, img_id in enumerate(ids):
                    # preds[i] shape: (C, H, W)
                    # Config.CLASS_LABELS order: large_bowel, small_bowel, stomach

                    for class_idx, class_name in enumerate(Config.CLASS_LABELS):
                        mask = preds[i, class_idx, :, :]

                        # Apply optional post-processing (e.g. min area)
                        if (
                            Config.MIN_MASK_AREA > 0
                            and np.sum(mask) < Config.MIN_MASK_AREA
                        ):
                            rle = ""
                        else:
                            rle = rle_encode(mask)

                        results.append(
                            {"id": img_id, "class": class_name, "predicted": rle}
                        )

        # Create DataFrame and save
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
