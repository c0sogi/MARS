import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import get_device, rle_encoding
from library.model import HybridSegFormerUNet
from library.dataset import InkDataset

# Ensure reproducibility
set_seed(Config.SEED)


class InferenceEngine:
    """
    Manages the inference process using Decoupled Volumetric Z-Scanning.
    """

    def __init__(self, checkpoint_path=None):
        """
        Args:
            checkpoint_path (str, optional): Path to the model checkpoint.
                                             Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        """
        self.device = get_device()
        self.model = HybridSegFormerUNet().to(self.device)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        if os.path.exists(checkpoint_path):
            print(f"Loading model weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization."
            )

        self.model.eval()

    def generate_submission(self):
        """
        Executes the Decoupled Z-Scanning inference strategy:
        1. Iterates through defined Z-offsets.
        2. Generates probability maps for each offset.
        3. Fuses maps using Maximum Probability Projection.
        4. Encodes results to RLE and saves submission.csv.
        """
        print("Starting Decoupled Z-Scanning Inference...")

        # 1. Initialize global probability maps for all test fragments
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA_PATH}"
            )

        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        fragment_maps = {}
        fragment_masks = {}  # To mask out invalid regions later

        print(f"Initializing maps for {len(test_df)} fragments...")
        for _, row in test_df.iterrows():
            fid = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Load binary mask to determine shape and valid regions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(
                    f"Failed to load mask for fragment {fid} at {mask_path}"
                )

            h, w = mask.shape
            fragment_maps[fid] = np.zeros((h, w), dtype=np.float32)
            fragment_masks[fid] = mask > 0

        # 2. Perform Scanning
        for offset in Config.SCAN_OFFSETS:
            z_start = Config.Z_START + offset
            print(f"Processing Scan: Z-Start {z_start} (Offset {offset})...")

            # Initialize dataset for this specific Z-slab
            # We use load_cached_data=True to leverage the caching mechanism in dataset.py
            dataset = InkDataset(
                Config.TEST_METADATA_PATH,
                mode="test",
                z_start=z_start,
                load_cached_data=True,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            with torch.no_grad():
                for images, coords, fids in dataloader:
                    images = images.to(self.device)

                    # Forward pass
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)  # (B, 1, H, W)

                    # Move to CPU
                    probs_np = probs.cpu().numpy()
                    coords_np = coords.numpy()

                    # Update global maps
                    for i in range(len(fids)):
                        fid = fids[i]
                        x, y = coords_np[i]
                        pred_tile = probs_np[i, 0]  # (H_tile, W_tile)

                        # Determine placement in global map
                        full_h, full_w = fragment_maps[fid].shape
                        tile_h, tile_w = pred_tile.shape

                        y_end = min(y + tile_h, full_h)
                        x_end = min(x + tile_w, full_w)

                        # Calculate valid dimensions (handling edge padding in dataset vs clipping here)
                        # InkDataset pads the input image if it's smaller than TILE_SIZE.
                        # The prediction will thus be TILE_SIZE x TILE_SIZE.
                        # We only want the part that corresponds to the valid fragment area.
                        valid_h = y_end - y
                        valid_w = x_end - x

                        # Extract valid region from prediction
                        pred_valid = pred_tile[:valid_h, :valid_w]

                        # Max-Fusion Update
                        # We take the maximum of the current accumulated value and the new prediction
                        current_val = fragment_maps[fid][y:y_end, x:x_end]
                        fragment_maps[fid][y:y_end, x:x_end] = np.maximum(
                            current_val, pred_valid
                        )

        # 3. Post-processing and RLE Encoding
        print("Finalizing predictions and generating submission...")
        submission_data = []

        for fid in sorted(fragment_maps.keys()):
            prob_map = fragment_maps[fid]
            valid_mask = fragment_masks[fid]

            # Thresholding
            binary_map = prob_map > Config.THRESHOLD

            # Apply valid mask (exclude predictions outside the fragment mask)
            binary_map = binary_map & valid_mask

            # Encode
            rle_str = rle_encoding(binary_map)
            submission_data.append({"Id": fid, "Predicted": rle_str})

        # 4. Save to CSV
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
