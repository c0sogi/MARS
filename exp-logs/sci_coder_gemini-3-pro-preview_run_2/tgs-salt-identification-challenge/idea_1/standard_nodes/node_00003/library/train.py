import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

from library.config import Config
from library.model import DepthLinkNet
from library.dataset import SaltDataset
from library.utils import set_seed, metric_iou, rle_encode


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice_loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred_sig.sum() + target.sum() + self.smooth)
        )

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class Trainer:
    """
    Manages the training, validation, and submission generation processes.
    """

    def __init__(self, config=None):
        self.config = config if config else Config()
        set_seed(self.config.SEED)
        self.device = torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = DepthLinkNet(
            in_channels=self.config.CHANNELS, num_classes=self.config.NUM_CLASSES
        ).to(self.device)

        # Loss Function
        self.criterion = BCEDiceLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS, eta_min=1e-6
        )

    def train(self, epochs=None, debug_limit=None):
        """
        Executes the training loop.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to config.EPOCHS.
            debug_limit (int, optional): Limit dataset size for debugging purposes.
        """
        run_epochs = epochs if epochs is not None else self.config.EPOCHS

        # Initialize Datasets with caching enabled
        train_ds = SaltDataset(
            self.config.TRAIN_CSV, self.config, mode="train", load_cached_data=True
        )
        val_ds = SaltDataset(
            self.config.VAL_CSV, self.config, mode="val", load_cached_data=True
        )

        # Apply debugging limit if specified
        if debug_limit is not None:
            train_ds = Subset(train_ds, range(min(len(train_ds), debug_limit)))
            val_ds = Subset(val_ds, range(min(len(val_ds), debug_limit)))

        # DataLoaders
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

        best_iou = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {run_epochs} epochs.")

        for epoch in range(run_epochs):
            self.model.train()
            train_loss = 0.0

            for images, depths, masks in train_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images, depths)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            val_loss, val_iou = self.validate(val_loader)
            self.scheduler.step()

            # Print metrics (Full precision for Val IoU)
            print(
                f"Epoch {epoch+1}/{run_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val IoU: {val_iou}"
            )

            # Checkpointing and Early Stopping
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_PATH)
                print(f"New Best Model Saved! IoU: {best_iou}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val IoU: {best_iou}")

    def validate(self, loader):
        """
        Runs validation on the given loader with TTA (Cite solution_lesson_node_00002).
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, depths, masks in loader:
                images = images.to(self.device)
                depths = depths.to(self.device)
                masks = masks.to(self.device)

                # TTA: Forward
                outputs = self.model(images, depths)

                # TTA: Flip
                images_flip = torch.flip(images, [3])
                outputs_flip = self.model(images_flip, depths)
                outputs_flip = torch.flip(outputs_flip, [3])

                # Average
                outputs_avg = (outputs + outputs_flip) / 2.0

                loss = self.criterion(outputs_avg, masks)
                total_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs_avg)
                batch_iou = metric_iou(masks, probs, threshold=self.config.THRESHOLD)
                total_iou += batch_iou * images.size(0)

        return total_loss / len(loader.dataset), total_iou / len(loader.dataset)

    def generate_submission(self):
        """
        Generates the submission file using the best trained model.
        """
        print("Generating submission...")

        # Load best weights
        if os.path.exists(self.config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(self.config.CHECKPOINT_PATH, map_location=self.device)
            )
            print("Loaded best checkpoint.")
        else:
            print("Warning: No checkpoint found, using current model weights.")

        self.model.eval()

        test_ds = SaltDataset(
            self.config.TEST_CSV, self.config, mode="test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        results = []

        # Calculate cropping indices to revert padding (128x128 -> 101x101)
        h, w = self.config.ORIG_SHAPE
        target_h, target_w = self.config.INPUT_SHAPE
        pad_top = (target_h - h) // 2
        pad_left = (target_w - w) // 2

        with torch.no_grad():
            for images, depths, ids in test_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)

                outputs = self.model(images, depths)
                probs = torch.sigmoid(outputs)
                probs = probs.cpu().numpy()

                for i, img_id in enumerate(ids):
                    prob_map = probs[i, 0, :, :]
                    # Crop center
                    mask_map = prob_map[pad_top : pad_top + h, pad_left : pad_left + w]

                    # Threshold
                    binary_mask = (mask_map > self.config.THRESHOLD).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(binary_mask)
                    results.append({"id": img_id, "rle_mask": rle})

        sub_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
