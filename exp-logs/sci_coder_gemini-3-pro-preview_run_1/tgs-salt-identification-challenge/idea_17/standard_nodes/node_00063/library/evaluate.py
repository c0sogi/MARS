import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, rle_encode, do_kaggle_metric
from library.model import DeepResUNet
from library.dataset import get_loaders, get_test_loader


class Evaluator:
    """
    Manages the evaluation, ensembling, and submission generation process.
    Implements the Gated Homogeneous Ensemble strategy with Test-Time Augmentation.
    """

    def __init__(self, debug=False):
        """
        Args:
            debug (bool): If True, uses a subset of data for faster debugging.
        """
        self.debug = debug
        self.device = Config.DEVICE

        # Ensure reproducibility
        seed_everything(Config.SEED)

        # Ensure directories exist
        Config.setup_directories()

        # Calculate crop indices to restore 101x101 from 128x128
        # Padding was reflection: Top=13, Bottom=14, Left=13, Right=14
        pad_h = Config.IMG_H - Config.ORIG_H
        pad_w = Config.IMG_W - Config.ORIG_W
        self.crop_top = pad_h // 2
        self.crop_bottom = self.crop_top + Config.ORIG_H
        self.crop_left = pad_w // 2
        self.crop_right = self.crop_left + Config.ORIG_W

    def load_model(self, checkpoint_path):
        """
        Loads a DeepResUNet model from a checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return None

        model = DeepResUNet().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_with_tta(self, model, images, depths):
        """
        Performs inference on a batch of images using Test-Time Augmentation (Horizontal Flip).
        """
        with torch.no_grad():
            # 1. Standard Forward Pass
            output = model(images, depths)
            probs = torch.sigmoid(output)

            # 2. TTA: Horizontal Flip
            if Config.TTA_FLIP:
                images_flipped = torch.flip(images, dims=[3])
                output_flipped = model(images_flipped, depths)
                probs_flipped = torch.sigmoid(output_flipped)

                # Flip predictions back to original orientation
                probs_flipped = torch.flip(probs_flipped, dims=[3])

                # Average standard and flipped predictions
                probs = (probs + probs_flipped) / 2.0

        return probs

    def validate_checkpoint(self, checkpoint_name, val_loader):
        """
        Calculates the validation mAP for a specific checkpoint to determine if it passes the quality gate.
        """
        path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)
        print(f"Validating checkpoint: {checkpoint_name}...")

        model = self.load_model(path)
        if model is None:
            return -1.0

        all_preds = []
        all_targets = []

        # Iterate over validation set
        for images, masks, depths, _ in val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            # Predict with TTA
            probs = self.predict_with_tta(model, images, depths)

            # Crop to original dimensions (101x101) for accurate metric calculation
            probs_cropped = probs[
                :, :, self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
            ]
            masks_cropped = masks[
                :, :, self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
            ]

            all_preds.append(probs_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())

        # Free memory
        del model
        torch.cuda.empty_cache()

        # Concatenate batches
        all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
        all_targets = np.concatenate(all_targets, axis=0).squeeze(1)

        # Calculate mAP
        score = do_kaggle_metric(all_preds, all_targets, threshold=0.5)
        return score

    def gated_ensemble(self):
        """
        Executes the Gated Homogeneous Ensemble pipeline:
        1. Evaluates Cycle 2, 3, and 4 checkpoints on the validation set.
        2. Selects checkpoints that are within the quality threshold of the best model.
        3. Generates ensemble predictions on the test set.
        4. Saves the submission file.
        """
        print("Starting Gated Ensemble Evaluation...")

        # --- Step 1: Validation and Gating ---
        # Load validation data
        _, val_loader = get_loaders(debug=self.debug, load_cached_data=True)

        # Dynamically select candidates from Cycle 2 onwards
        candidates = [f"best_cycle_{i}.pth" for i in range(2, Config.CYCLES + 1)]
        scores = {}

        # Compute scores for each candidate
        for cp in candidates:
            score = self.validate_checkpoint(cp, val_loader)
            if score >= 0:
                scores[cp] = score
                print(f"Checkpoint {cp} | Validation mAP: {score:.10f}")

        if not scores:
            print("Error: No valid checkpoints found. Cannot generate submission.")
            return

        # Determine best score and threshold
        best_score = max(scores.values())
        threshold = best_score - Config.QUALITY_GATE_THRESHOLD

        selected_checkpoints = []
        print(f"Applying Quality Gate (Threshold: {threshold:.10f})...")

        for cp, score in scores.items():
            if score >= threshold:
                selected_checkpoints.append(cp)
                print(f"  [ACCEPTED] {cp} (Score: {score:.10f})")
            else:
                print(f"  [REJECTED] {cp} (Score: {score:.10f} < Threshold)")

        # --- Step 2: Test Inference ---
        print("Loading selected models for inference...")
        models = []
        for cp in selected_checkpoints:
            path = os.path.join(Config.CHECKPOINT_DIR, cp)
            models.append(self.load_model(path))

        print("Generating predictions on test set...")
        test_loader = get_test_loader(load_cached_data=True)

        submission_rows = []

        for images, _, depths, ids in test_loader:
            images = images.to(self.device)
            depths = depths.to(self.device)

            # Ensemble Averaging
            avg_probs = None
            for model in models:
                probs = self.predict_with_tta(model, images, depths)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(models)

            # Crop to original size (101x101)
            avg_probs = avg_probs[
                :, :, self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
            ]

            # Move to CPU
            avg_probs_np = avg_probs.cpu().numpy().squeeze(1)

            # Encode predictions
            for i in range(len(ids)):
                img_id = ids[i]
                # Binarize at 0.5 threshold
                pred_mask = (avg_probs_np[i] > 0.5).astype(np.uint8)
                rle = rle_encode(pred_mask)
                submission_rows.append([img_id, rle])

        # --- Step 3: Save Submission ---
        df_sub = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
