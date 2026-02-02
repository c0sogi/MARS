import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.model import SaltUNetPlusPlus
from library.utils import rle_encode, calculate_iou_map, seed_everything
from library.dataset import make_loader, get_stratified_folds


class InferenceRunner:
    """
    Handles inference tasks including Test-Time Augmentation (TTA),
    Out-Of-Fold (OOF) prediction generation, global threshold optimization,
    and final submission generation using a stratified ensemble.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.cache_dir = Config.IDEA_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_model(self, fold_idx):
        """
        Loads the best checkpoint for a specific fold.

        Args:
            fold_idx (int): The fold index (0-4).

        Returns:
            nn.Module: Loaded model in eval mode, or None if checkpoint is missing.
        """
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found at {model_path}")
            return None

        model = SaltUNetPlusPlus()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _predict_tta(self, model, images):
        """
        Performs inference with Test-Time Augmentation (Horizontal Flip).

        Args:
            model (nn.Module): The trained model.
            images (torch.Tensor): Input batch (B, C, H, W).

        Returns:
            torch.Tensor: Predicted probability maps resized to original 101x101 resolution.
        """
        # 1. Forward Pass: Original
        with torch.cuda.amp.autocast():
            out_orig = model(images)  # Output is (B, 1, 128, 128)
            prob_orig = torch.sigmoid(out_orig)

        # 2. Forward Pass: Horizontal Flip
        # Flip width dimension (dim 3)
        images_flip = torch.flip(images, dims=[3])
        with torch.cuda.amp.autocast():
            out_flip = model(images_flip)
            prob_flip = torch.sigmoid(out_flip)

        # Revert flip on predictions
        prob_flip = torch.flip(prob_flip, dims=[3])

        # 3. Average Predictions
        prob_avg = (prob_orig + prob_flip) / 2.0

        # 4. Resize to Original Resolution (101x101)
        # Use bilinear interpolation for probabilities
        prob_resized = F.interpolate(
            prob_avg,
            size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
            mode="bilinear",
            align_corners=True,
        )

        return prob_resized

    def generate_oof_predictions(self, load_cached=True):
        """
        Generates or loads Out-Of-Fold predictions for the entire training set.
        Used to optimize the binarization threshold.

        Args:
            load_cached (bool): If True, attempts to load results from disk.

        Returns:
            tuple: (all_preds, all_gts) numpy arrays of shape (N, 101, 101).
        """
        cache_path_preds = os.path.join(self.cache_dir, "cached_oof_preds.npy")
        cache_path_gts = os.path.join(self.cache_dir, "cached_oof_gts.npy")

        # 1. Try Loading from Cache
        if load_cached:
            if os.path.exists(cache_path_preds) and os.path.exists(cache_path_gts):
                print("Loading OOF predictions from cache...")
                try:
                    all_preds = np.load(cache_path_preds)
                    all_gts = np.load(cache_path_gts)
                    return all_preds, all_gts
                except Exception as e:
                    print(f"Failed to load OOF cache: {e}. Recomputing...")

        # 2. Compute OOF Predictions
        print("Generating OOF predictions from scratch...")
        folds = get_stratified_folds(n_splits=Config.NUM_FOLDS)

        all_preds_list = []
        all_gts_list = []

        for fold_idx, (_, val_df) in enumerate(folds):
            print(f"Processing Fold {fold_idx} OOF...")
            model = self._load_model(fold_idx)

            if model is None:
                print(f"Skipping Fold {fold_idx} (Model not found).")
                continue

            # Create validation loader for this fold
            # Use a unique cache name to avoid conflicts with training caches
            val_loader = make_loader(
                val_df,
                phase="val",
                batch_size=Config.BATCH_SIZE,
                load_cached=True,
                cache_name=f"val_fold_{fold_idx}_inference",
                shuffle=False,
            )

            fold_preds = []
            fold_gts = []

            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images = images.to(self.device)

                    # Predict with TTA
                    probs = self._predict_tta(model, images)  # (B, 1, 101, 101)

                    # Resize Ground Truth to 101x101 if necessary
                    if masks.shape[-2:] != (Config.ORIG_HEIGHT, Config.ORIG_WIDTH):
                        masks = F.interpolate(
                            masks.float(),
                            size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                            mode="nearest",
                        )

                    fold_preds.append(probs.cpu().numpy())
                    fold_gts.append(masks.cpu().numpy())

            # Aggregate fold results
            if fold_preds:
                all_preds_list.append(np.concatenate(fold_preds, axis=0))
                all_gts_list.append(np.concatenate(fold_gts, axis=0))

            # Cleanup memory
            del model
            torch.cuda.empty_cache()

        # Concatenate all folds
        if all_preds_list:
            all_preds = np.concatenate(all_preds_list, axis=0)  # (N, 1, 101, 101)
            all_gts = np.concatenate(all_gts_list, axis=0)  # (N, 1, 101, 101)

            # Squeeze channel dimension
            all_preds = all_preds.squeeze(1)
            all_gts = all_gts.squeeze(1)

            # 3. Save to Cache
            np.save(cache_path_preds, all_preds)
            np.save(cache_path_gts, all_gts)
        else:
            print("No predictions generated. Returning empty arrays.")
            all_preds = np.array([])
            all_gts = np.array([])

        return all_preds, all_gts

    def optimize_threshold(self, preds, gts):
        """
        Sweeps through binarization thresholds to find the one that maximizes mAP.

        Args:
            preds (np.ndarray): Probability maps (N, 101, 101).
            gts (np.ndarray): Ground truth masks (N, 101, 101).

        Returns:
            float: The optimal threshold.
        """
        if len(preds) == 0:
            print(
                "No predictions provided for threshold optimization. Defaulting to 0.5."
            )
            return 0.5

        print("Optimizing binarization threshold...")
        # Sweep from 0.3 to 0.7
        thresholds = np.linspace(0.3, 0.7, 41)
        best_score = -1.0
        best_th = 0.5

        # Ensure GT is binary (0 or 1)
        gts = (gts > 0.5).astype(np.uint8)

        for th in thresholds:
            # Binarize predictions
            binary_preds = (preds > th).astype(np.uint8)

            # Calculate mAP
            score = calculate_iou_map(binary_preds, gts, verbose=False)

            if score > best_score:
                best_score = score
                best_th = th

        print(f"Best Threshold: {best_th} | Best OOF mAP: {best_score}")
        return best_th

    def generate_submission(self, threshold=0.5):
        """
        Generates the final submission file by ensembling all trained models
        and applying Test-Time Augmentation.

        Args:
            threshold (float): The binarization threshold to use.
        """
        print(f"Generating submission with threshold {threshold}...")

        # Load Test Metadata
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Create Test Loader
        test_loader = make_loader(
            test_df,
            phase="test",
            batch_size=Config.BATCH_SIZE,
            load_cached=True,
            cache_name="test_inference",
            shuffle=False,
        )

        # Load all available models
        models = []
        for i in range(Config.NUM_FOLDS):
            m = self._load_model(i)
            if m is not None:
                models.append(m)

        if not models:
            print("No models loaded. Cannot generate submission.")
            return

        results = []

        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(self.device)

                # Accumulator for ensemble probabilities
                avg_prob = torch.zeros(
                    (images.size(0), 1, Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                    device=self.device,
                )

                # Aggregate predictions from all models
                for model in models:
                    # Predict with TTA (returns 101x101 resized probs)
                    probs = self._predict_tta(model, images)
                    avg_prob += probs

                # Average
                avg_prob /= len(models)

                # Move to CPU for post-processing
                avg_prob = avg_prob.cpu().numpy().squeeze(1)  # (B, 101, 101)

                # Process batch
                for i, img_id in enumerate(ids):
                    prob_map = avg_prob[i]

                    # Binarize
                    mask_bin = (prob_map > threshold).astype(np.uint8)

                    # Encode
                    rle = rle_encode(mask_bin)
                    results.append((img_id, rle))

        # Save Submission
        sub_df = pd.DataFrame(results, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
