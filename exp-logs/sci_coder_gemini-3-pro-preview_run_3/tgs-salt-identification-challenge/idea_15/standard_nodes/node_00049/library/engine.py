import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import calc_map, rle_encode
from library.losses import BCEDiceLoss, LovaszHingeLoss


class SaltEngine:
    """
    Engine for training, validating, and predicting with the Salt Segmentation model.
    Implements Dynamic Deep Supervision and Phase-based training curriculum.
    """

    def __init__(self, model, optimizer, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # Loss Functions
        self.criterion_phase1 = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
        self.criterion_phase2 = LovaszHingeLoss(per_image=True)

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self, loader, epoch):
        """
        Runs one epoch of training.
        Handles Phase switching and Deep Supervision toggling.
        """
        self.model.train()

        # Determine Phase
        # Phase 1: Epochs 0 to 19 (First 20 epochs)
        # Phase 2: Epochs 20+
        is_phase1 = epoch < Config.PHASE1_EPOCHS

        # Toggle Deep Supervision in Model
        # The model class checks this flag during forward pass if self.training is True
        self.model.deep_supervision = is_phase1

        running_loss = 0.0
        dataset_size = 0

        # Loop over batches
        for images, masks, _ in loader:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(images)

                loss = 0.0
                if is_phase1:
                    # Phase 1: Deep Supervision Active
                    # Model returns list: [logit1, logit2, logit3, final_logit]
                    # We apply BCE+Dice to all outputs with equal weight
                    if isinstance(outputs, list):
                        # Cite solution_lesson_node_00038: Average auxiliary losses to maintain consistent gradient scale.
                        for logits in outputs:
                            loss += self.criterion_phase1(logits, masks)
                        loss = loss / len(outputs)
                    else:
                        # Fallback safety
                        loss = self.criterion_phase1(outputs, masks)
                else:
                    # Phase 2: Deep Supervision Disabled
                    # Model returns single final_logit
                    # We apply Lovasz-Hinge Loss
                    loss = self.criterion_phase2(outputs, masks)

            # Backward pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        print(
            f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} | Phase {'1' if is_phase1 else '2'} | Train Loss: {epoch_loss:.6f}"
        )
        return epoch_loss

    def validate_one_epoch(self, loader):
        """
        Runs validation on the provided loader.
        Handles cropping to original resolution before metric calculation.
        Cite solution_lesson_node_00011: Performs dynamic threshold optimization.
        """
        self.model.eval()

        # Store all predictions and targets for global threshold optimization
        all_probs = []
        all_masks = []

        # Calculate cropping indices to revert 128x128 padding -> 101x101
        pad_total = Config.IMG_HEIGHT - Config.ORIG_HEIGHT
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        start_idx = pad_top
        end_idx = Config.IMG_HEIGHT - pad_bottom

        with torch.no_grad():
            for images, masks, _ in loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Crop predictions
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]

                # Handle masks
                if masks.ndim == 3:
                    masks_cropped = masks[:, start_idx:end_idx, start_idx:end_idx]
                else:
                    masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

                all_probs.append(probs_cropped.cpu().numpy())
                all_masks.append(masks_cropped.cpu().numpy())

        # Concatenate
        all_probs = np.concatenate(all_probs, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Dynamic Threshold Sweep
        thresholds = np.arange(0.3, 0.75, 0.05)
        best_val_map = 0.0

        for t in thresholds:
            # Shift probabilities to simulate thresholding at 0.5
            # prob > t <=> (prob - t + 0.5) > 0.5
            shifted_probs = np.clip(all_probs - t + 0.5, 0, 1)
            score = calc_map(shifted_probs, all_masks)
            if score > best_val_map:
                best_val_map = score

        print(f"Validation mAP (Optimized): {best_val_map}")
        return best_val_map

    def predict(self, loader, tta=True):
        """
        Generates predictions for the test set.
        Applies Test-Time Augmentation (Horizontal Flip) and Cropping.

        Returns:
            predictions (list of np.ndarray): List of 101x101 probability maps.
            ids (list of str): Corresponding image IDs.
        """
        self.model.eval()

        predictions = []
        ids_list = []

        # Crop indices
        pad_total = Config.IMG_HEIGHT - Config.ORIG_HEIGHT
        pad_top = pad_total // 2
        start_idx = pad_top
        end_idx = Config.IMG_HEIGHT - (pad_total - pad_top)

        with torch.no_grad():
            for images, ids in loader:
                images = images.to(self.device, dtype=torch.float32)

                # 1. Forward Pass (Original)
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                if tta:
                    # 2. Forward Pass (Flipped)
                    # Flip width dimension (dim 3)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = self.model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)

                    # Flip back to original orientation
                    probs_flipped = torch.flip(probs_flipped, dims=[3])

                    # Average
                    probs = (probs + probs_flipped) / 2.0

                # Crop to 101x101
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]

                # Convert to numpy
                # Remove channel dim: (B, 1, 101, 101) -> (B, 101, 101)
                preds_np = probs_cropped.cpu().numpy()[:, 0, :, :]

                for i in range(len(ids)):
                    predictions.append(preds_np[i])
                    ids_list.append(ids[i])

        return predictions, ids_list

    def save_predictions(self, predictions, ids, output_path, threshold=0.5):
        """
        Encodes predictions to RLE and saves to CSV.
        """
        rle_masks = []
        for pred in predictions:
            # Thresholding
            mask = (pred > threshold).astype(np.uint8)
            rle = rle_encode(mask)
            rle_masks.append(rle)

        df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
