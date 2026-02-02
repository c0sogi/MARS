import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import (
    DEVICE,
    Z_SCAN_VALUES,
    SUBMISSION_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    SEED,
)
from library.utils import rle_encode, calc_map, unpad_image, set_seed
from library.losses import TeacherComboLoss, StableBCELoss, DepthMSELoss
from library.dataset import get_depth_stats


class SaltEngine:
    def __init__(
        self, model, device=DEVICE, optimizer=None, scheduler=None, mode="teacher"
    ):
        """
        Args:
            model: The PyTorch model (SaltNet).
            device: 'cuda' or 'cpu'.
            optimizer: PyTorch optimizer.
            scheduler: Learning rate scheduler.
            mode: 'teacher' (Stage 1) or 'student' (Stage 3).
        """
        set_seed(SEED)
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode

        # Loss Functions
        self.loss_combo = TeacherComboLoss()
        self.loss_bce = StableBCELoss()
        self.loss_mse = DepthMSELoss()

    def train_one_epoch(self, dataloader, epoch_idx):
        """
        Trains the model for one epoch.
        Handles mixed batches (Labeled/Unlabeled) and Multi-Task learning.
        """
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            depths = batch["depth"].to(self.device)

            if self.optimizer:
                self.optimizer.zero_grad()

            loss = 0.0

            if self.mode == "teacher":
                # Teacher: Explicit Depth Injection
                logits = self.model(images, depths)
                loss = self.loss_combo(logits, masks)

            elif self.mode == "student":
                # Student: Image Only -> Seg + Aux Depth
                logits, pred_depth = self.model(images)

                # --- 1. Segmentation Loss ---
                # Identify soft targets (Pseudo-labels) vs Hard targets (GT)
                # Heuristic: Soft targets have values strictly between 0 and 1
                # We check if any pixel in the mask is 'soft'
                is_soft = (
                    ((masks > 0.0) & (masks < 1.0)).view(masks.size(0), -1).any(dim=1)
                )

                hard_indices = torch.nonzero(~is_soft).squeeze(1)
                soft_indices = torch.nonzero(is_soft).squeeze(1)

                loss_seg = 0.0

                # Hard Targets (GT): Use Combo Loss (Lovasz + BCE)
                if len(hard_indices) > 0:
                    l_hard = self.loss_combo(logits[hard_indices], masks[hard_indices])
                    loss_seg += l_hard * (len(hard_indices) / len(masks))

                # Soft Targets (Pseudo): Use Stable BCE
                if len(soft_indices) > 0:
                    l_soft = self.loss_bce(logits[soft_indices], masks[soft_indices])
                    loss_seg += l_soft * (len(soft_indices) / len(masks))

                # --- 2. Auxiliary Depth Loss ---
                # Only calculate MSE where ground truth depth is valid (not NaN)
                # Depths are (B, 1)
                valid_depth_mask = ~torch.isnan(depths).view(-1)

                loss_depth = 0.0
                if valid_depth_mask.any():
                    loss_depth = self.loss_mse(
                        pred_depth[valid_depth_mask], depths[valid_depth_mask]
                    )

                loss = loss_seg + loss_depth

            # Backward pass (FP32)
            loss.backward()

            if self.optimizer:
                self.optimizer.step()

            running_loss += loss.item()

        # Step Scheduler
        if self.scheduler:
            self.scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch_idx} | Mode: {self.mode} | Loss: {epoch_loss:.8f}")
        return epoch_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Optimizes the binarization threshold for mAP.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                depths = batch["depth"].to(self.device)

                if self.mode == "teacher":
                    logits = self.model(images, depths)
                else:
                    # Student ignores aux depth output for validation metrics
                    logits, _ = self.model(images)

                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Threshold Optimization
        # Linear search for the threshold that maximizes mAP
        best_map = 0.0
        best_thresh = 0.5
        thresholds = np.arange(0.1, 0.95, 0.05)

        for t in thresholds:
            # Binarize predictions
            binary_preds = (all_preds > t).float()
            score = calc_map(binary_preds, all_targets)

            if score > best_map:
                best_map = score
                best_thresh = t

        print(f"Validation mAP: {best_map:.8f} | Best Threshold: {best_thresh:.2f}")
        return best_map, best_thresh

    @staticmethod
    def predict_marginalized(model, image, device):
        """
        Performs marginalized inference by scanning across a range of depth values.
        Used in Stage 2 to generate robust pseudo-labels.
        """
        model.eval()
        image = image.to(device)

        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)

        accumulated_probs = torch.zeros_like(image[:, 0:1, :, :])

        with torch.no_grad():
            for z_val in Z_SCAN_VALUES:
                # Create depth tensor (B, 1) with the scan value
                # z_val is already in standard units (e.g., -1.5, 0, 1.5)
                z_tensor = torch.full(
                    (image.size(0), 1), z_val, dtype=torch.float32, device=device
                )

                logits = model(image, z_tensor)
                probs = torch.sigmoid(logits)
                accumulated_probs += probs

        # Average the probabilities
        avg_probs = accumulated_probs / len(Z_SCAN_VALUES)
        return avg_probs

    def generate_submission_csv(self, dataloader, threshold=0.5):
        """
        Generates predictions for the test set and saves to CSV.
        Applies Test-Time Augmentation (Horizontal Flip).
        """
        self.model.eval()
        ids = []
        rle_masks = []

        print("Generating submission...")

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                batch_ids = batch["id"]

                # 1. Original Inference
                if self.mode == "teacher":
                    # Fallback if using teacher (not recommended by plan, but handled)
                    # Use mean depth (0.0) if actual depth unknown/NaN
                    z_zero = torch.zeros((images.size(0), 1), device=self.device)
                    logits_orig = self.model(images, z_zero)
                else:
                    logits_orig, _ = self.model(images)

                probs_orig = torch.sigmoid(logits_orig)

                # 2. TTA: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])  # Flip width (dim 3)

                if self.mode == "teacher":
                    z_zero = torch.zeros((images.size(0), 1), device=self.device)
                    logits_flipped = self.model(images_flipped, z_zero)
                else:
                    logits_flipped, _ = self.model(images_flipped)

                probs_flipped = torch.sigmoid(logits_flipped)
                probs_flipped_back = torch.flip(probs_flipped, dims=[3])

                # Average Probabilities
                avg_probs = (probs_orig + probs_flipped_back) / 2.0

                # Binarize
                binary_preds = (avg_probs > threshold).float().cpu().numpy()

                for i, pred_mask in enumerate(binary_preds):
                    # Remove channel dim: (1, 128, 128) -> (128, 128)
                    pred_mask = pred_mask.squeeze(0)

                    # Unpad to original size (101, 101)
                    pred_mask = unpad_image(pred_mask)

                    # Encode
                    rle = rle_encode(pred_mask)

                    ids.append(batch_ids[i])
                    rle_masks.append(rle)

        # Create DataFrame and Save
        df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
        df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
