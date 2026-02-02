import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from library.config import (
    PathConfig,
    DataConfig,
    GeneralConfig,
    ModelConfig,
)
from library.utils import rle_encode, calculate_iou_map
from library.dataset import get_loaders, get_test_loader
from library.model import SaltUNetPlusPlus


class Evaluator:
    """
    Handles evaluation, threshold optimization, and submission generation.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(GeneralConfig.DEVICE)
        self.crop_start = (DataConfig.IMG_H - DataConfig.ORIG_H) // 2
        self.crop_end = self.crop_start + DataConfig.ORIG_H

    def _crop_prediction(self, tensor):
        """Crops the padded tensor (128x128) back to original size (101x101)."""
        # tensor shape: (B, 1, H, W) or (B, H, W)
        if tensor.dim() == 4:
            return tensor[
                :, :, self.crop_start : self.crop_end, self.crop_start : self.crop_end
            ]
        elif tensor.dim() == 3:
            return tensor[
                :, self.crop_start : self.crop_end, self.crop_start : self.crop_end
            ]
        return tensor

    def predict_fold(self, fold_idx, model_path, load_cached_data=True, debug=False):
        """
        Generates predictions for a specific fold's validation set.
        Includes caching mechanism for OOF predictions.
        """
        cache_file = os.path.join(PathConfig.CACHE_DIR, f"oof_pred_fold_{fold_idx}.npz")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached OOF predictions for Fold {fold_idx}...")
            data = np.load(cache_file)
            return data["preds"], data["targets"]

        # 2. Compute from scratch
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load Data
        _, val_loader = get_loaders(fold_idx, debug=debug)

        # Load Model
        model = SaltUNetPlusPlus()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(self.device)

                # Forward Pass (Original)
                out = model(images)
                prob = torch.sigmoid(out)

                # Test-Time Augmentation (Horizontal Flip)
                images_flipped = torch.flip(images, dims=[3])
                out_flipped = model(images_flipped)
                prob_flipped = torch.sigmoid(out_flipped)
                prob_flipped = torch.flip(prob_flipped, dims=[3])

                # Average
                avg_prob = (prob + prob_flipped) / 2.0

                # Crop to original size
                avg_prob = self._crop_prediction(avg_prob)
                masks = self._crop_prediction(masks)

                preds_list.append(avg_prob.cpu().numpy())
                targets_list.append(masks.cpu().numpy())

        preds_arr = np.concatenate(preds_list, axis=0)
        targets_arr = np.concatenate(targets_list, axis=0)

        # Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, preds=preds_arr, targets=targets_arr)

        return preds_arr, targets_arr

    def optimize_threshold(self, preds, targets):
        """
        Finds the best binarization threshold by maximizing the competition metric (mAP).

        Args:
            preds (np.array): Probabilities (N, 1, H, W) or (N, H, W).
            targets (np.array): Binary masks (N, 1, H, W) or (N, H, W).

        Returns:
            float: Optimal threshold.
        """
        print("Optimizing binarization threshold...")

        thresholds = np.linspace(0.3, 0.7, 41)  # Sweep from 0.30 to 0.70
        best_score = -1.0
        best_threshold = 0.5

        # Ensure correct shapes for metric calculation
        if preds.ndim == 4:
            preds = preds.squeeze(1)
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        for th in thresholds:
            # Binarize
            binary_preds = (preds > th).astype(np.uint8)

            # Calculate Metric (mAP over IoU thresholds 0.5:0.95)
            # We use the utility function which handles the IoU sweep internally
            score = calculate_iou_map(
                binary_preds, targets, threshold=0.5
            )  # threshold arg here is dummy since input is already binary

            if score > best_score:
                best_score = score
                best_threshold = th

        print(f"Optimal Threshold: {best_threshold:.4f} (mAP: {best_score:.5f})")
        return best_threshold

    def generate_submission(self, model_paths, threshold=0.5, debug=False):
        """
        Generates the final submission CSV using an ensemble of models.
        """
        print(f"Generating submission with ensemble of {len(model_paths)} models...")
        print(f"Using threshold: {threshold}")

        test_loader = get_test_loader(debug=debug)

        # We process batch by batch, accumulating predictions from all models
        # This saves memory compared to loading all models at once if models are huge,
        # though with 40GB GPU we could likely load all. We stick to a safe approach.

        # Pre-load models to avoid reloading weights for every batch (Speed vs Memory trade-off)
        # With 5 ResNeXt50 models, this fits in A100 memory easily.
        loaded_models = []
        for path in model_paths:
            m = SaltUNetPlusPlus()
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.to(self.device)
            m.eval()
            loaded_models.append(m)

        predictions = {}

        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(self.device)
                batch_size = images.size(0)

                # Accumulator for ensemble probabilities
                ensemble_preds = torch.zeros(
                    (batch_size, 1, DataConfig.IMG_H, DataConfig.IMG_W),
                    device=self.device,
                )

                for model in loaded_models:
                    # Original
                    out = model(images)
                    prob = torch.sigmoid(out)
                    ensemble_preds += prob

                    # TTA (Flip)
                    images_flipped = torch.flip(images, dims=[3])
                    out_flipped = model(images_flipped)
                    prob_flipped = torch.sigmoid(out_flipped)
                    prob_flipped = torch.flip(prob_flipped, dims=[3])
                    ensemble_preds += prob_flipped

                # Average
                # (Num_Models * 2 views)
                ensemble_preds /= len(loaded_models) * 2

                # Crop to 101x101
                ensemble_preds = self._crop_prediction(ensemble_preds)

                # Binarize
                binary_preds = (ensemble_preds > threshold).byte().cpu().numpy()

                # Encode
                for i, img_id in enumerate(ids):
                    mask = binary_preds[i, 0]  # (H, W)
                    rle = rle_encode(mask)
                    predictions[img_id] = rle

        # Create DataFrame
        sub_df = pd.DataFrame.from_dict(
            predictions, orient="index", columns=["rle_mask"]
        )
        sub_df.index.name = "id"
        sub_df.reset_index(inplace=True)

        # Save
        output_path = os.path.join(PathConfig.SUBMISSION_DIR, "submission.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

        return sub_df

    def run_full_evaluation(self, model_paths, debug=False):
        """
        Helper to run the full OOF generation -> Threshold Optimization -> Submission pipeline.
        """
        all_oof_preds = []
        all_oof_targets = []

        # 1. Generate OOF for all folds
        for fold_idx, path in enumerate(model_paths):
            if not os.path.exists(path):
                print(
                    f"Warning: Model for fold {fold_idx} not found at {path}. Skipping."
                )
                continue

            p, t = self.predict_fold(fold_idx, path, debug=debug)
            all_oof_preds.append(p)
            all_oof_targets.append(t)

        if not all_oof_preds:
            raise ValueError("No OOF predictions generated. Check model paths.")

        # Concatenate
        full_preds = np.concatenate(all_oof_preds, axis=0)
        full_targets = np.concatenate(all_oof_targets, axis=0)

        # 2. Optimize Threshold
        best_threshold = self.optimize_threshold(full_preds, full_targets)

        # 3. Generate Submission
        self.generate_submission(model_paths, threshold=best_threshold, debug=debug)
