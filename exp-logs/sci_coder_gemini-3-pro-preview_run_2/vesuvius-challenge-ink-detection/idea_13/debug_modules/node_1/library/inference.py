import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import SiameseSegFormer
from library.data import VesuviusDataset
from library.utils import rle_encode


class InferenceRunner:
    """
    Manages the inference pipeline for the Siamese Multi-View SegFormer.
    Handles TTA, patch stitching, and submission generation.
    """

    def __init__(self, checkpoint_path=Config.CHECKPOINT_PATH):
        self.device = torch.device(Config.DEVICE)
        self.checkpoint_path = checkpoint_path

        # Initialize model architecture
        self.model = SiameseSegFormer().to(self.device)

        # Load weights
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print(
                f"Warning: Checkpoint not found at {self.checkpoint_path}. Inference may fail."
            )
            return

        print(f"Loading model weights from {self.checkpoint_path}...")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _predict_batch(self, x_high, x_center, x_low):
        """
        Helper to run forward pass and apply sigmoid.
        """
        logits = self.model(x_high, x_center, x_low)
        return torch.sigmoid(logits)

    def run(self):
        """
        Executes the full inference pipeline:
        1. Load Test Data
        2. TTA Inference (Original, H-Flip, V-Flip, Rot90)
        3. Stitching patches into full fragment maps
        4. RLE Encoding and CSV generation
        """
        print("Initializing Inference Pipeline...")

        # 1. Setup Data Loader
        test_ds = VesuviusDataset(mode="test", transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Setup Fragment Canvases
        # We need to reconstruct the full image from patches.
        # We'll use the test metadata to get the dimensions of each fragment.
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

        fragment_preds = {}
        fragment_counts = {}
        fragment_masks_path = {}

        for _, row in test_meta_df.iterrows():
            frag_id = str(row["fragment_id"])
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            fragment_masks_path[frag_id] = mask_path

            # Read mask to get dimensions
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue

            h, w = mask_img.shape
            fragment_preds[frag_id] = np.zeros((h, w), dtype=np.float32)
            fragment_counts[frag_id] = np.zeros((h, w), dtype=np.float32)

        # 3. Inference Loop with TTA
        print("Running inference with TTA...")
        dataset_df = test_ds.df

        with torch.no_grad():
            for x_high, x_center, x_low, indices in test_loader:
                x_high = x_high.to(self.device)
                x_center = x_center.to(self.device)
                x_low = x_low.to(self.device)

                # --- TTA Strategy ---
                # 1. Original
                pred = self._predict_batch(x_high, x_center, x_low)

                # 2. Horizontal Flip (dim 3)
                x_high_h = torch.flip(x_high, [3])
                x_center_h = torch.flip(x_center, [3])
                x_low_h = torch.flip(x_low, [3])
                pred_h = self._predict_batch(x_high_h, x_center_h, x_low_h)
                pred += torch.flip(pred_h, [3])

                # 3. Vertical Flip (dim 2)
                x_high_v = torch.flip(x_high, [2])
                x_center_v = torch.flip(x_center, [2])
                x_low_v = torch.flip(x_low, [2])
                pred_v = self._predict_batch(x_high_v, x_center_v, x_low_v)
                pred += torch.flip(pred_v, [2])

                # 4. Rotate 90 (k=1, dims=[2,3])
                x_high_r = torch.rot90(x_high, 1, [2, 3])
                x_center_r = torch.rot90(x_center, 1, [2, 3])
                x_low_r = torch.rot90(x_low, 1, [2, 3])
                pred_r = self._predict_batch(x_high_r, x_center_r, x_low_r)
                # Inverse is Rotate 270 (k=3)
                pred += torch.rot90(pred_r, 3, [2, 3])

                # Average
                pred /= 4.0

                # --- Stitching ---
                pred_np = pred.cpu().numpy()  # (B, 1, H, W)
                indices_np = indices.numpy().flatten()

                for i, idx in enumerate(indices_np):
                    # Get patch metadata
                    row = dataset_df.iloc[idx]
                    frag_id = str(row["fragment_id"])
                    x, y = int(row["x"]), int(row["y"])
                    w, h = int(row["width"]), int(row["height"])

                    # Extract prediction for this patch
                    patch_pred = pred_np[i, 0, :, :]  # (H, W)

                    # Determine placement dimensions (handle potential edge cases)
                    # The patch prediction is always TILE_SIZE x TILE_SIZE (or smaller if input was smaller)
                    ph, pw = patch_pred.shape

                    # Add to canvas
                    fragment_preds[frag_id][y : y + ph, x : x + pw] += patch_pred
                    fragment_counts[frag_id][y : y + ph, x : x + pw] += 1.0

        # 4. Generate Submission
        print("Generating submission file...")
        submission_data = []

        for frag_id in fragment_preds:
            # Average the predictions
            counts = fragment_counts[frag_id]
            # Avoid division by zero
            counts[counts == 0] = 1.0

            prob_map = fragment_preds[frag_id] / counts

            # Load original mask to ensure validity
            mask_path = fragment_masks_path[frag_id]
            valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Thresholding
            binary_map = (prob_map > 0.5).astype(np.uint8)

            # Apply valid mask
            if valid_mask is not None:
                binary_map = binary_map * (valid_mask > 0)

            # RLE Encode
            rle_str = rle_encode(binary_map)
            submission_data.append({"Id": frag_id, "Predicted": rle_str})

        # Save to CSV
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
