import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, rle_encode, do_kaggle_metric
from library.model import SaltLinkNet
from library.losses import BCEDiceLoss
from library.dataset import get_dataloaders


class SaltTrainer:
    """
    Trainer class for Salt Segmentation task.
    Handles training, validation, threshold optimization, and inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = SaltLinkNet().to(self.device)

        # Loss Function
        self.criterion = BCEDiceLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Tracking
        self.best_score = -float("inf")
        self.best_loss = float("inf")

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(loader.dataset)

        for images, masks, depths in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images, depths)
            loss = self.criterion(logits, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / dataset_size

    def validate(self, loader, threshold=0.5):
        """
        Runs validation and calculates metric.
        """
        self.model.eval()
        running_loss = 0.0
        preds_list = []
        masks_list = []
        dataset_size = len(loader.dataset)

        with torch.no_grad():
            for images, masks, depths in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                logits = self.model(images, depths)
                loss = self.criterion(logits, masks)
                running_loss += loss.item() * images.size(0)

                # Apply sigmoid for metric calculation
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                masks_list.append(masks.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        preds_all = np.concatenate(preds_list, axis=0)
        masks_all = np.concatenate(masks_list, axis=0)

        # Calculate mAP metric
        score = do_kaggle_metric(preds_all, masks_all, threshold=threshold)

        return epoch_loss, score

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        patience = Config.EARLY_STOPPING_PATIENCE
        min_delta = Config.EARLY_STOPPING_MIN_DELTA
        counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader, threshold=0.5)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Score (mAP@0.5): {val_score:.6f}"
            )

            # Checkpoint and Early Stopping
            if val_score > self.best_score + min_delta:
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # print(f"Model saved. New best score: {self.best_score:.6f}")
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def load_best_model(self):
        """Loads the best saved model state."""
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )

    def optimize_threshold(self, loader):
        """
        Finds the best binarization threshold on the validation set.
        """
        self.load_best_model()
        self.model.eval()

        preds_list = []
        masks_list = []

        with torch.no_grad():
            for images, masks, depths in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                logits = self.model(images, depths)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                masks_list.append(masks.cpu().numpy())

        preds_all = np.concatenate(preds_list, axis=0)
        masks_all = np.concatenate(masks_list, axis=0)

        best_thr = 0.5
        best_score = -1.0

        # Sweep thresholds from 0.3 to 0.75
        thresholds = np.arange(0.3, 0.76, 0.05)

        for thr in thresholds:
            score = do_kaggle_metric(preds_all, masks_all, threshold=thr)
            if score > best_score:
                best_score = score
                best_thr = thr

        print(
            f"Optimal Threshold found: {best_thr:.2f} with Validation Score: {best_score:.6f}"
        )
        return best_thr

    def predict(self, test_loader, threshold=0.5):
        """
        Generates predictions for the test set using TTA and saves submission.
        """
        self.load_best_model()
        self.model.eval()

        submission_data = []

        # Calculate cropping indices to revert padding
        # Original: 101, Target: 128. Diff: 27.
        # Pad logic: Top=13, Bottom=14, Left=13, Right=14
        h_start = 13
        h_end = 13 + Config.IMG_ORIG_SIZE
        w_start = 13
        w_end = 13 + Config.IMG_ORIG_SIZE

        # Load Test IDs for mapping
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ids = test_df["id"].values

        idx_counter = 0

        with torch.no_grad():
            for images, depths in test_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)

                # 1. Forward Pass (Original)
                logits = self.model(images, depths)
                probs = torch.sigmoid(logits)

                # 2. TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = self.model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                # Flip predictions back
                probs_flip = torch.flip(probs_flip, dims=[3])

                # 3. Average Predictions
                avg_probs = (probs + probs_flip) / 2.0
                avg_probs = avg_probs.cpu().numpy()  # (B, 1, 128, 128)

                # 4. Process Batch
                for i in range(avg_probs.shape[0]):
                    # Crop center to 101x101
                    pred_mask = avg_probs[i, 0, h_start:h_end, w_start:w_end]

                    # Binarize
                    binary_mask = (pred_mask > threshold).astype(np.uint8)

                    # Encode
                    rle = rle_encode(binary_mask)

                    # Store
                    current_id = test_ids[idx_counter]
                    submission_data.append([current_id, rle])
                    idx_counter += 1

        # Save Submission
        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_pipeline():
    """
    Orchestrates the full training, validation, optimization, and submission pipeline.
    """
    # Reproducibility
    set_seed(Config.SEED)
    Config.setup()

    # Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Trainer
    trainer = SaltTrainer()

    # Train
    trainer.fit(train_loader, val_loader)

    # Optimize Threshold
    best_threshold = trainer.optimize_threshold(val_loader)

    # Predict and Submit
    trainer.predict(test_loader, threshold=best_threshold)
