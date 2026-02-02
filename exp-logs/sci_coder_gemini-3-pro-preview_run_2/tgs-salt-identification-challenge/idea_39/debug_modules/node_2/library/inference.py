import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.utils import rle_encode, do_kaggle_metric, get_logger


class InferenceEngine:
    """
    Handles inference logic including Threshold Optimization and Marginalized Depth-Scan.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = get_logger(Config.WORKING_DIR)

    def _crop_to_original(self, prob_map):
        """
        Crops the padded 128x128 probability map back to the original 101x101 size.
        Assumes Center Padding was used during preprocessing.
        """
        h, w = prob_map.shape[-2:]
        th, tw = Config.ORIG_SIZE, Config.ORIG_SIZE

        if h == th and w == tw:
            return prob_map

        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return prob_map[..., start_h : start_h + th, start_w : start_w + tw]

    def predict_val(self, val_loader):
        """
        Runs inference on the validation set using TRUE depths.
        This is used to determine the optimal binarization threshold.
        """
        self.model.eval()
        all_probs = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                # Validation loader yields: images, masks, depths, ids
                images, masks, depths, ids = batch

                images = images.to(self.device, dtype=torch.float32)
                depths = depths.to(self.device, dtype=torch.float32)

                # Forward pass using the specific depth
                outputs = self.model(images, depths)

                # Handle potential tuple output (logits, aux)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                # Crop back to original size
                probs = self._crop_to_original(probs)

                # Store results
                all_probs.append(probs.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

        return np.concatenate(all_probs), np.concatenate(all_masks)

    def optimize_threshold(self, val_loader):
        """
        Finds the probability threshold that maximizes the mAP score on the validation set.
        """
        self.logger.info("Optimizing binarization threshold on validation set...")

        # Get predictions and ground truth
        probs, true_masks = self.predict_val(val_loader)

        best_threshold = 0.5
        best_score = -1.0

        # Sweep range: 0.30 to 0.75
        thresholds = np.arange(0.3, 0.76, 0.05)

        for t in thresholds:
            # Calculate mAP at this binarization threshold
            # do_kaggle_metric handles the internal IoU sweeping
            score = do_kaggle_metric(probs, true_masks, threshold=t)
            self.logger.info(f"Threshold: {t:.2f} | Validation mAP: {score:.10f}")

            if score > best_score:
                best_score = score
                best_threshold = t

        self.logger.info(
            f"Best Threshold found: {best_threshold:.2f} with mAP: {best_score:.10f}"
        )
        return best_threshold

    def predict_scan(self, test_loader, scan_depths=None):
        """
        Runs inference on the test set using Marginalized Depth-Scan.
        Averages predictions across multiple depth hypotheses and Horizontal Flip TTA.
        """
        if scan_depths is None:
            scan_depths = Config.SCAN_DEPTHS

        self.logger.info(f"Starting Inference with Scan Depths: {scan_depths}")
        self.model.eval()

        all_ids = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                # Test loader yields: images, ids
                # (Dataset logic ensures only 2 items for test mode)
                if len(batch) == 2:
                    images, ids = batch
                else:
                    # Fallback
                    images = batch[0]
                    ids = batch[-1]

                images = images.to(self.device, dtype=torch.float32)
                batch_size = images.size(0)

                # Container for accumulating probabilities
                # Shape: (B, 1, 128, 128)
                batch_avg_probs = torch.zeros(
                    (batch_size, 1, Config.IMG_SIZE, Config.IMG_SIZE),
                    device=self.device,
                    dtype=torch.float32,
                )

                # Prepare TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])

                total_passes = 0

                for z_val in scan_depths:
                    # Create constant depth tensor for this scan step
                    z_tensor = torch.full(
                        (batch_size, 1), z_val, device=self.device, dtype=torch.float32
                    )

                    # 1. Forward pass: Original Image
                    out = self.model(images, z_tensor)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    batch_avg_probs += torch.sigmoid(logits)
                    total_passes += 1

                    # 2. Forward pass: Flipped Image (TTA)
                    out_flip = self.model(images_flip, z_tensor)
                    logits_flip = (
                        out_flip[0] if isinstance(out_flip, (tuple, list)) else out_flip
                    )
                    probs_flip = torch.sigmoid(logits_flip)
                    # Flip back to original orientation
                    batch_avg_probs += torch.flip(probs_flip, dims=[3])
                    total_passes += 1

                # Compute average
                batch_avg_probs /= total_passes

                # Crop to original size (101x101)
                batch_avg_probs = self._crop_to_original(batch_avg_probs)

                # Move to CPU and store
                all_probs.append(batch_avg_probs.cpu().numpy())
                all_ids.extend(ids)

        return np.concatenate(all_probs), all_ids

    def generate_submission(self, test_loader, threshold=0.5):
        """
        Orchestrates the submission generation process.
        """
        self.logger.info(
            f"Generating submission with binary threshold: {threshold:.4f}"
        )

        # Get marginalized predictions
        probs, ids = self.predict_scan(test_loader)

        # Binarize predictions
        preds = (probs > threshold).astype(np.uint8)

        rle_list = []
        for i in range(len(ids)):
            # Extract mask for current image
            mask = preds[i]
            # Remove channel dim if present (C, H, W) -> (H, W)
            if mask.ndim == 3:
                mask = mask[0]

            # Encode
            rle = rle_encode(mask)
            rle_list.append(rle)

        # Create DataFrame
        sub_df = pd.DataFrame({"id": ids, "rle_mask": rle_list})

        # Save to disk
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        return sub_df
