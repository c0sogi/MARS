import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.model import DeepResUNet
from library.dataset import SaltDataset
from library.utils import rle_encode


class Predictor:
    """
    Handles inference and submission generation for the Salt Segmentation task.
    Implements Hybrid Snapshot Ensembling (Cycle 2 Best + SWA) with Test-Time Augmentation.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Calculate cropping indices to revert padding (128 -> 101)
        # Config.IMG_SIZE = 128, Config.ORIG_SIZE = 101
        pad_total = Config.IMG_SIZE - Config.ORIG_SIZE
        self.pad_top = pad_total // 2
        self.pad_bottom = pad_total - self.pad_top
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left

    def load_model(self, checkpoint_name):
        """
        Loads a DeepResUNet model from the checkpoint directory.

        Args:
            checkpoint_name (str): Name of the .pth file in the checkpoint directory.

        Returns:
            model (nn.Module): Loaded model in eval mode, or None if file missing.
        """
        path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found.")
            return None

        model = DeepResUNet()

        try:
            # Load state dict
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading state dict for {checkpoint_name}: {e}")
            return None

        model.to(self.device)
        model.eval()
        return model

    def predict(self, use_tta=True):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            use_tta (bool): Whether to use Test-Time Augmentation (Horizontal Flip).
        """
        # 1. Load Data
        print("Initializing Test Dataset...")
        test_dataset = SaltDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Load Models for Ensemble
        # Strategy: Ensemble Best Cycle 2 (Dice) and SWA (Lovasz)
        checkpoint_names = ["best_cycle_2.pth", "swa_model.pth"]
        models = []

        print("Loading models for ensemble...")
        for name in checkpoint_names:
            model = self.load_model(name)
            if model is not None:
                models.append(model)

        # Fallback if specific checkpoints are missing (e.g., if training was shorter than expected)
        if not models:
            print(
                "Specific ensemble checkpoints not found. Falling back to 'best_model.pth'."
            )
            model = self.load_model("best_model.pth")
            if model is not None:
                models.append(model)

        if not models:
            raise RuntimeError(f"No valid checkpoints found in {Config.CHECKPOINT_DIR}")

        print(f"Ensembling {len(models)} model(s).")

        # 3. Inference Loop
        results = []

        with torch.no_grad():
            for i, (images, _, ids) in enumerate(test_loader):
                images = images.to(self.device)

                # Accumulate probabilities from all models
                batch_probs_sum = None

                for model in models:
                    # Forward Pass (Original)
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                    if use_tta:
                        # Forward Pass (Horizontal Flip)
                        images_flip = torch.flip(images, dims=[3])
                        logits_flip = model(images_flip)
                        probs_flip = torch.sigmoid(logits_flip)

                        # Un-flip predictions
                        probs_flip_back = torch.flip(probs_flip, dims=[3])

                        # Average TTA
                        probs = (probs + probs_flip_back) / 2.0

                    if batch_probs_sum is None:
                        batch_probs_sum = probs
                    else:
                        batch_probs_sum += probs

                # Average over ensemble
                avg_probs = batch_probs_sum / len(models)

                # 4. Post-processing
                # Crop padding: (B, 1, 128, 128) -> (B, 1, 101, 101)
                # Slicing: [..., top:bottom, left:right]
                # Note: Config.IMG_SIZE is 128.
                # The valid region is from pad_top to (IMG_SIZE - pad_bottom)
                cropped_probs = avg_probs[
                    ...,
                    self.pad_top : Config.IMG_SIZE - self.pad_bottom,
                    self.pad_left : Config.IMG_SIZE - self.pad_right,
                ]

                # Remove channel dimension: (B, 101, 101)
                cropped_probs = cropped_probs.squeeze(1)

                # Thresholding
                binary_masks = (cropped_probs > 0.5).byte().cpu().numpy()

                # RLE Encoding
                for idx, img_id in enumerate(ids):
                    mask = binary_masks[idx]
                    rle = rle_encode(mask)
                    results.append({"id": img_id, "rle_mask": rle})

                if (i + 1) % 10 == 0:
                    print(f"Processed batch {i + 1}/{len(test_loader)}")

        # 5. Save Submission
        print("Saving submission...")
        df = pd.DataFrame(results)

        # Ensure correct column order
        df = df[["id", "rle_mask"]]

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
