import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_iou_batch, rle_encode
from library.losses import BCEDiceLoss, LovaszHingeLoss, SoftBCELoss


class SaltEngine:
    """
    Handles training, validation, and inference for the Salt Segmentation task.
    Implements the Stabilized Semi-Supervised U-Net++ Ensemble strategy.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = amp.GradScaler()

        # Loss Functions
        self.bce_dice_loss = BCEDiceLoss()
        self.lovasz_loss = LovaszHingeLoss()
        self.soft_bce_loss = SoftBCELoss()

    def get_loss_fn(self, epoch, phase2=False):
        """
        Selects the appropriate loss function based on the training curriculum.
        """
        if phase2:
            # Phase 2: Soft-Target Self-Training
            return self.soft_bce_loss

        # Phase 1: Supervised Training Curriculum
        if epoch <= Config.LOSS_SWITCH_EPOCH:
            return self.bce_dice_loss
        else:
            return self.lovasz_loss

    def train_one_epoch(self, loader, epoch, phase2=False):
        """
        Runs one epoch of training with AMP and Deep Supervision.
        """
        self.model.train()
        running_loss = 0.0
        criterion = self.get_loss_fn(epoch, phase2)

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with amp.autocast():
                # Deep Supervision: Model returns a list of outputs
                outputs = self.model(images)

                # Calculate loss for each auxiliary head
                loss = 0.0
                for output in outputs:
                    loss += criterion(output, targets)

                # Normalize loss by number of heads if necessary,
                # but strategy implies equal weights summing to higher gradient signal.
                # We stick to sum as per "Equal Weights (1.0)".

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Runs validation and calculates mAP.
        """
        self.model.eval()
        running_loss = 0.0
        running_map = 0.0

        # Use Lovasz for validation loss as it correlates best with IoU
        criterion = self.lovasz_loss

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with amp.autocast():
                # Model returns single tensor in eval mode
                output = self.model(images)
                loss = criterion(output, targets)

            running_loss += loss.item()

            # Metric Calculation
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(output)

            # calculate_iou_batch handles cropping internally if shapes mismatch
            batch_map = calculate_iou_batch(probs, targets, threshold=0.5)
            running_map += batch_map

        avg_loss = running_loss / len(loader)
        avg_map = running_map / len(loader)

        return avg_loss, avg_map

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        save_path="best_model.pth",
        phase2=False,
        patience=10,
    ):
        """
        Full training loop with Early Stopping and Checkpointing.
        """
        best_map = 0.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs (Phase 2: {phase2})...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch, phase2=phase2)
            val_loss, val_map = self.evaluate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Val mAP: {val_map:.8f}"
            )

            # Scheduler Step (Monitor mAP)
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_map)
                else:
                    self.scheduler.step()

            # Checkpointing & Early Stopping
            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                # print(f"New best model saved with mAP: {best_map:.8f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        print(f"Training finished. Best mAP: {best_map:.8f}")

    @torch.no_grad()
    def predict_test_set(
        self, loader, threshold=0.5, output_path=Config.SUBMISSION_PATH
    ):
        """
        Generates predictions for the test set using TTA and saves to CSV.
        """
        self.model.eval()
        ids_list = []
        rle_list = []

        print("Generating predictions with TTA...")

        for images, ids in loader:
            images = images.to(self.device)

            # TTA 1: Original
            with amp.autocast():
                out_orig = self.model(images)
                probs_orig = torch.sigmoid(out_orig)

            # TTA 2: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            with amp.autocast():
                out_flip = self.model(images_flipped)
                probs_flip = torch.sigmoid(out_flip)

            # Flip back the predictions
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average Predictions
            probs_avg = (probs_orig + probs_flip_back) / 2.0

            # Crop from 128x128 (padded) to 101x101 (original)
            if probs_avg.shape[-1] != Config.ORIG_SIZE:
                h, w = probs_avg.shape[-2:]
                start_h = (h - Config.ORIG_SIZE) // 2
                start_w = (w - Config.ORIG_SIZE) // 2
                probs_avg = probs_avg[
                    :,
                    :,
                    start_h : start_h + Config.ORIG_SIZE,
                    start_w : start_w + Config.ORIG_SIZE,
                ]

            # Binarize
            preds_bin = (probs_avg > threshold).float().cpu().numpy()

            # Encode
            for i in range(len(ids)):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds_bin[i, 0, :, :]
                rle = rle_encode(mask)
                ids_list.append(ids[i])
                rle_list.append(rle)

        # Save Submission
        df = pd.DataFrame({"id": ids_list, "rle_mask": rle_list})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
