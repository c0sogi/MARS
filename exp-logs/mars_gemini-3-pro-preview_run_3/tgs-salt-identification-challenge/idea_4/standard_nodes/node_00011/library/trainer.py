import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calc_map, optimize_thresholds, rle_encode
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.dataset import SaltDataset
from library.model import SaltModel


class Trainer:
    """
    Manages training, validation, and submission generation for the Salt Segmentation task.
    """

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.SEED)

        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = SaltModel(config).to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Reduce LR if validation metric plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )

        # Losses
        self.bce_dice_loss = BCEDiceLoss()
        self.lovasz_loss = LovaszHingeLoss()

        # State tracking
        self.best_score = 0.0
        self.best_threshold = 0.5

    def _crop_to_original(self, tensor_or_array):
        """
        Crops the padded 128x128 output back to 101x101.
        Padding logic from dataset.py:
        128 - 101 = 27. Top=13, Bottom=14, Left=13, Right=14.
        """
        # Slicing indices
        h_start, h_end = 13, 128 - 14
        w_start, w_end = 13, 128 - 14

        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array[..., h_start:h_end, w_start:w_end]
        else:
            return tensor_or_array[..., h_start:h_end, w_start:w_end]

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0

        # Loss Switching Logic
        if epoch < self.config.LOVASZ_SWITCH_EPOCH:
            criterion = self.bce_dice_loss
            loss_name = "BCE+Dice"
        else:
            criterion = self.lovasz_loss
            loss_name = "Lovasz-Hinge"

        for images, masks, _ in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            logits = self.model(images)

            # Calculate Loss
            loss = criterion(logits, masks)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        return epoch_loss, loss_name

    def validate(self, loader, return_probs=False):
        self.model.eval()
        running_loss = 0.0
        criterion = self.bce_dice_loss  # Use consistent loss for reporting val_loss

        all_probs = []
        all_masks = []

        with torch.no_grad():
            for images, masks, _ in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = criterion(logits, masks)
                running_loss += loss.item() * images.size(0)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Crop back to original size (101x101) for accurate metric calculation
                probs_cropped = self._crop_to_original(probs)
                masks_cropped = self._crop_to_original(masks)

                all_probs.append(probs_cropped.cpu().numpy())
                all_masks.append(masks_cropped.cpu().numpy())

        val_loss = running_loss / len(loader.dataset)

        # Concatenate
        probs_arr = np.concatenate(all_probs, axis=0).squeeze(1)  # (N, 101, 101)
        masks_arr = np.concatenate(all_masks, axis=0).squeeze(1)  # (N, 101, 101)

        # Calculate mAP
        score = calc_map(masks_arr, probs_arr, threshold=0.5)

        if return_probs:
            return val_loss, score, probs_arr, masks_arr
        return val_loss, score

    def fit(self):
        # 1. Load Data
        train_df = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(self.config.VAL_METADATA_PATH)

        train_ds = SaltDataset(train_df, mode="train", config=self.config)
        val_ds = SaltDataset(val_df, mode="val", config=self.config)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training for {self.config.EPOCHS} epochs...")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss, loss_name = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_score)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} [{loss_name}] - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val mAP: {val_score:.10f} - "
                f"Time: {elapsed:.2f}s"
            )

            # Save Best Model
            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(
                    self.model.state_dict(),
                    self.config.get_model_save_path("best_model.pth"),
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping Logic
            # Reset patience if we just switched losses
            if epoch == self.config.LOVASZ_SWITCH_EPOCH:
                print("Loss switched to Lovasz-Hinge. Resetting patience.")
                patience_counter = 0

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val mAP: {self.best_score:.10f}")

        # 2. Optimize Threshold
        self.find_optimal_threshold(val_loader)

        # 3. Generate Submission
        self.generate_submission()

    def find_optimal_threshold(self, val_loader):
        print("Loading best model for threshold optimization...")
        self.model.load_state_dict(
            torch.load(self.config.get_model_save_path("best_model.pth"))
        )

        print("Gathering validation probabilities...")
        _, _, probs, masks = self.validate(val_loader, return_probs=True)

        print("Optimizing threshold...")
        self.best_threshold = optimize_thresholds(masks, probs, verbose=True)

    def generate_submission(self):
        print("Generating submission for test set...")

        # Load Test Data
        test_df = pd.read_csv(self.config.TEST_METADATA_PATH)
        test_ds = SaltDataset(test_df, mode="test", config=self.config)
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensure model is best state
        self.model.load_state_dict(
            torch.load(self.config.get_model_save_path("best_model.pth"))
        )
        self.model.eval()

        predictions = []
        ids = []

        with torch.no_grad():
            for images, _, batch_ids in test_loader:
                images = images.to(self.device)

                # 1. Original Prediction
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # 2. TTA: Horizontal Flip
                # Flip input on width dimension (dim 3)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = self.model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)
                # Flip output back
                probs_flipped_back = torch.flip(probs_flipped, dims=[3])

                # Average predictions
                avg_probs = (probs + probs_flipped_back) / 2.0

                # Crop to 101x101
                avg_probs_cropped = self._crop_to_original(avg_probs)

                # Binarize using optimal threshold
                pred_masks = (avg_probs_cropped > self.best_threshold).float()

                # Convert to numpy
                pred_masks_np = pred_masks.cpu().numpy().squeeze(1)  # (B, 101, 101)

                for i in range(len(batch_ids)):
                    rle = rle_encode(pred_masks_np[i])
                    predictions.append(rle)
                    ids.append(batch_ids[i])

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": ids, "rle_mask": predictions})
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
