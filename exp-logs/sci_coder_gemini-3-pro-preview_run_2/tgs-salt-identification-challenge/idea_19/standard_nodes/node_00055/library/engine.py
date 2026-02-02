import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import itertools
from library.utils import set_seed, calculate_map, do_unpad, rle_encode
from library.losses import CombinedLoss


class SaltEngine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        """
        Initializes the training/inference engine.

        Args:
            model: The PyTorch model (ResNet34WideLinkNet).
            device: torch.device ('cuda' or 'cpu').
            optimizer: torch.optim optimizer (optional).
            scheduler: Learning rate scheduler (optional).
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        # BCEWithLogitsLoss is used for the consistency loss in Noisy Student
        self.consistency_loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader, loss_fn):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, masks, depths, ids in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images, depths)
            loss = loss_fn(logits, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, loader, loss_fn):
        """
        Evaluates the model on the validation set.
        Computes Loss and mAP with threshold optimization.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_gts = []

        with torch.no_grad():
            for images, masks, depths, ids in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                # Forward
                logits = self.model(images, depths)
                loss = loss_fn(logits, masks)
                total_loss += loss.item()
                num_batches += 1

                # Prepare for mAP calculation
                # Sigmoid to get probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                gts = masks.cpu().numpy()

                # Unpad and accumulate
                for i in range(len(probs)):
                    # Extract single image (C, H, W) -> (128, 128)
                    p = probs[i][0]
                    g = gts[i][0]

                    # Unpad back to 101x101
                    p_unpad = do_unpad(p, original_shape=(101, 101))
                    g_unpad = do_unpad(g, original_shape=(101, 101))

                    all_preds.append(p_unpad)
                    all_gts.append(g_unpad)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Threshold search for best mAP
        best_map = 0.0
        best_thresh = 0.5
        thresholds = np.arange(0.3, 0.75, 0.05)

        for t in thresholds:
            binary_preds = [(p > t).astype(np.uint8) for p in all_preds]
            score = calculate_map(binary_preds, all_gts)
            if score > best_map:
                best_map = score
                best_thresh = t

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation mAP: {best_map:.4f} (at threshold {best_thresh:.2f})")

        return avg_loss, best_map, best_thresh

    def predict_proba(self, loader):
        """
        Generates probability maps for the test set using Test-Time Augmentation (TTA).
        TTA: Average of original and horizontally flipped predictions.

        Returns:
            dict: {id: probability_map_numpy_array (101x101)}
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for batch in loader:
                # Handle variable unpacking (Test vs Val loaders)
                if len(batch) == 3:
                    images, depths, ids = batch
                elif len(batch) == 4:
                    images, _, depths, ids = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                images = images.to(self.device)
                depths = depths.to(self.device)

                # TTA: Create flipped version
                images_flip = torch.flip(images, dims=[3])  # Flip width

                # Predict
                logits = self.model(images, depths)
                logits_flip = self.model(images_flip, depths)

                # Sigmoid
                probs = torch.sigmoid(logits)
                probs_flip = torch.sigmoid(logits_flip)

                # Revert flip
                probs_flip = torch.flip(probs_flip, dims=[3])

                # Average
                avg_probs = (probs + probs_flip) / 2.0
                avg_probs = avg_probs.cpu().numpy()

                # Unpad and store
                for i, img_id in enumerate(ids):
                    p = avg_probs[i][0]  # (128, 128)
                    p_unpad = do_unpad(p, original_shape=(101, 101))
                    results[img_id] = p_unpad

        return results

    def generate_submission(self, loader, output_path, threshold=0.5):
        """
        Generates the submission CSV file.

        Args:
            loader: Test data loader.
            output_path: Path to save the CSV.
            threshold: Probability threshold for binarization.
        """
        print("Generating predictions with TTA...")
        predictions = self.predict_proba(loader)

        print(f"Encoding masks with threshold {threshold}...")
        submission_rows = []

        # Sort by ID to ensure consistent order (though not strictly required by CSV)
        sorted_ids = sorted(predictions.keys())

        for img_id in sorted_ids:
            prob_map = predictions[img_id]

            # Binarize
            binary_mask = (prob_map > threshold).astype(np.uint8)

            # RLE Encode
            rle = rle_encode(binary_mask)

            submission_rows.append({"id": img_id, "rle_mask": rle})

        # Create DataFrame and save
        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
