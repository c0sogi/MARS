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

    def train_teacher_epoch(self, loader, loss_fn):
        """
        Trains the teacher model for one epoch.
        Bernoulli Depth Masking is handled by the dataset/loader configuration.
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

    def train_student_epoch(self, labeled_loader, unlabeled_loader, supervised_loss_fn):
        """
        Trains the student model for one epoch using Noisy Student logic.

        Args:
            labeled_loader: Loader for ground truth data.
            unlabeled_loader: Loader for test data with soft pseudo-labels.
            supervised_loss_fn: Loss function for labeled data.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Handle different loader lengths by cycling the shorter one (usually labeled)
        # or cycling unlabeled if labeled is longer (unlikely here).
        if len(labeled_loader) < len(unlabeled_loader):
            loader_zip = zip(itertools.cycle(labeled_loader), unlabeled_loader)
            steps = len(unlabeled_loader)
        else:
            loader_zip = zip(labeled_loader, itertools.cycle(unlabeled_loader))
            steps = len(labeled_loader)

        for (l_images, l_masks, l_depths, _), (
            u_images,
            u_targets,
            u_depths,
            _,
        ) in loader_zip:
            # Move data to device
            l_images = l_images.to(self.device)
            l_masks = l_masks.to(self.device)
            l_depths = l_depths.to(self.device)

            u_images = u_images.to(self.device)
            u_targets = u_targets.to(self.device)  # Soft targets (probs)
            u_depths = u_depths.to(self.device)

            self.optimizer.zero_grad()

            # --- Supervised Step ---
            l_logits = self.model(l_images, l_depths)
            l_loss = supervised_loss_fn(l_logits, l_masks)

            # --- Unsupervised (Consistency) Step ---
            # Student predicts on strongly augmented unlabeled images
            u_logits = self.model(u_images, u_depths)
            # Loss against Teacher's soft targets
            u_loss = self.consistency_loss_fn(u_logits, u_targets)

            # Combine losses
            loss = l_loss + u_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / steps if steps > 0 else 0.0
        return avg_loss

    def validate(self, loader, loss_fn):
        """
        Evaluates the model on the validation set.
        Computes Loss and mAP.
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

        # Calculate mAP
        # Note: calculate_map internally binarizes predictions at 0.5 for IoU calculation
        map_score = calculate_map(all_preds, all_gts)

        print(f"Validation Loss: {avg_loss}")
        print(f"Validation mAP: {map_score}")

        return avg_loss, map_score

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
            for images, depths, ids in loader:
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
