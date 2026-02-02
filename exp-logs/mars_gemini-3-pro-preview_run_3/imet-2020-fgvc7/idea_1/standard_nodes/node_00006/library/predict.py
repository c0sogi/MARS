import os
import torch
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from library.config import Config
from library.model import ArtworkClassifier
from library.utils import calculate_micro_f1


class Predictor:
    """
    Handles inference, threshold optimization, and submission generation for the
    Multi-Label Artwork Classification task.
    """

    def __init__(self, device=None):
        """
        Initialize the Predictor with model and device.
        """
        self.config = Config
        self.device = device if device else torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = ArtworkClassifier(
            num_classes=self.config.NUM_CLASSES, pretrained=False
        )
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path=None):
        """
        Loads model weights from the specified checkpoint path.
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.MODEL_SAVE_PATH

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {checkpoint_path}")

    def _run_inference(self, loader):
        """
        Internal helper to run inference on a dataloader.
        Returns probabilities and targets (if available).
        """
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)

                # Use Automatic Mixed Precision for inference
                with autocast(enabled=self.config.USE_AMP):
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return all_probs, all_targets

    def optimize_threshold(self, val_loader):
        """
        Finds the best threshold based on validation predictions to maximize Micro F1.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.

        Returns:
            float: The optimized threshold.
        """
        print("Running inference on validation set for threshold optimization...")
        probs, targets = self._run_inference(val_loader)

        # Range of thresholds to test
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0.0
        best_thresh = self.config.DEFAULT_THRESHOLD

        for thresh in thresholds:
            score = calculate_micro_f1(probs, targets, threshold=thresh)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        print(f"Threshold Optimization Results:")
        print(f"Best Threshold: {best_thresh}")
        print(f"Best Validation Micro F1: {best_f1}")  # Printing full precision

        return best_thresh

    def generate_submission(self, test_loader, threshold):
        """
        Generates predictions for the test set using the specified threshold
        and saves them to the submission CSV.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            threshold (float): Threshold to binarize probabilities.
        """
        print(f"Generating submission using threshold: {threshold}...")

        # Run inference
        probs, _ = self._run_inference(test_loader)

        # Binarize predictions
        preds = (probs >= threshold).astype(int)

        # Retrieve IDs from the dataset
        # Note: We assume the loader is not shuffled (shuffle=False in get_dataloaders)
        test_ids = test_loader.dataset.data["id"].values

        submission_rows = []
        for idx, row in enumerate(preds):
            image_id = test_ids[idx]
            # Get indices of active attributes (where value is 1)
            attr_indices = np.where(row == 1)[0]
            # Join indices with spaces
            attr_str = " ".join(map(str, attr_indices))
            submission_rows.append({"id": image_id, "attribute_ids": attr_str})

        # Create DataFrame and save
        submission_df = pd.DataFrame(submission_rows)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
