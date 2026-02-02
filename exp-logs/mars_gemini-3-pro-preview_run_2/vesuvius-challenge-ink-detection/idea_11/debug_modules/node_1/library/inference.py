import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import rle_encoding, sigmoid
from library.model import SegFormer
from library.dataset import prepare_volumes, InkDataset


class InferenceEngine:
    def __init__(self, model_path=None, device=None):
        """
        Initializes the Inference Engine.
        Args:
            model_path (str): Path to the trained model checkpoint.
            device (str): Device to run inference on.
        """
        self.device = device if device else Config.DEVICE
        self.model = SegFormer().to(self.device)

        # Load Model Weights
        if model_path is None:
            model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"InferenceEngine: Loaded model from {model_path}")
        else:
            print(
                f"InferenceEngine: Warning - Model not found at {model_path}. Using random initialization."
            )

        self.model.eval()

    def _predict_batch_tta(self, images):
        """
        Predicts a batch of images using Test Time Augmentation (TTA).
        TTA Strategy: Original + Horizontal Flip + Vertical Flip.
        Returns: Averaged probability map.
        """
        # 1. Original
        logits = self.model(images)
        probs = torch.sigmoid(logits)

        # 2. Flip Horizontal
        images_flip_h = torch.flip(images, dims=[3])
        logits_h = self.model(images_flip_h)
        probs_h = torch.sigmoid(logits_h)
        probs_h = torch.flip(probs_h, dims=[3])

        # 3. Flip Vertical
        images_flip_v = torch.flip(images, dims=[2])
        logits_v = self.model(images_flip_v)
        probs_v = torch.sigmoid(logits_v)
        probs_v = torch.flip(probs_v, dims=[2])

        # Average predictions
        avg_probs = (probs + probs_h + probs_v) / 3.0
        return avg_probs

    def predict_fragment(self, fragment_id, load_cached_data=True):
        """
        Performs Decoupled Z-Scanning inference on a single fragment.

        Args:
            fragment_id (str): The ID of the fragment to predict.
            load_cached_data (bool): Whether to use cached volumes.

        Returns:
            np.ndarray: The fused probability map for the fragment.
        """
        # 1. Locate Data and Determine Dimensions
        # Try test directory first, then train (for debugging/validation)
        base_dir = os.path.join(Config.INPUT_DIR, "test", fragment_id)
        if not os.path.exists(base_dir):
            base_dir = os.path.join(Config.INPUT_DIR, "train", fragment_id)

        mask_path = os.path.join(base_dir, "mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for fragment {fragment_id}")

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h_img, w_img = mask_img.shape

        # 2. Generate Tiling Metadata
        # We manually generate patches to ensure we cover the whole image
        patches = []
        for y in range(0, h_img, Config.TILE_SIZE):
            for x in range(0, w_img, Config.TILE_SIZE):
                patches.append(
                    {
                        "fragment_id": fragment_id,
                        "x": x,
                        "y": y,
                        "width": Config.TILE_SIZE,
                        "height": Config.TILE_SIZE,
                        "mask_path": os.path.relpath(mask_path, Config.INPUT_DIR),
                    }
                )
        df_tiles = pd.DataFrame(patches)

        # 3. Prepare 3D Volume (Cached)
        volumes = prepare_volumes([fragment_id], load_cached_data=load_cached_data)

        # 4. Initialize Max-Fusion Canvas
        final_fused_prob = np.zeros((h_img, w_img), dtype=np.float32)

        # 5. Decoupled Z-Scanning Loop
        # Iterate through defined depth offsets to capture wandering ink
        for z_offset in Config.INFERENCE_Z_OFFSETS:
            current_z = Config.Z_START + z_offset
            # print(f"  Scanning Z={current_z}...")

            # Create Dataset for this specific Z-depth
            ds = InkDataset(df_tiles, volumes, z_start=current_z, mode="test")
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Canvas for the current Z-pass
            z_prob_map = np.zeros((h_img, w_img), dtype=np.float32)

            with torch.no_grad():
                for images, _, _, indices in loader:
                    images = images.to(self.device)

                    # Predict with TTA
                    probs = self._predict_batch_tta(images)
                    probs = probs.cpu().numpy()  # Shape: (B, 1, H, W)

                    # Map predictions back to the canvas
                    current_indices = indices.numpy()
                    for i, idx in enumerate(current_indices):
                        row = df_tiles.iloc[idx]
                        x, y = row["x"], row["y"]

                        # Extract prediction
                        pred_patch = probs[i, 0, :, :]

                        # Handle boundaries (un-pad if necessary)
                        # Calculate valid region within the image bounds
                        y_end = min(y + Config.TILE_SIZE, h_img)
                        x_end = min(x + Config.TILE_SIZE, w_img)

                        valid_h = y_end - y
                        valid_w = x_end - x

                        # Crop the prediction to the valid area
                        pred_valid = pred_patch[:valid_h, :valid_w]

                        # Assign to Z-map
                        z_prob_map[y:y_end, x:x_end] = pred_valid

            # Max-Fusion: Update the final map with the maximum probability found so far
            final_fused_prob = np.maximum(final_fused_prob, z_prob_map)

        # 6. Apply Fragment Mask
        # Ensure predictions are strictly within the valid fragment area
        mask_binary = (mask_img > 0).astype(np.float32)
        final_fused_prob = final_fused_prob * mask_binary

        return final_fused_prob


def z_scan_predict(load_cached_data=True):
    """
    Main entry point for generating the submission file.
    Executes the inference pipeline on all test fragments.
    """
    # 1. Load Test Metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        print("Error: Test metadata file not found.")
        return

    df_test_meta = pd.read_csv(test_csv_path)
    fragment_ids = df_test_meta["fragment_id"].unique()

    # 2. Initialize Engine
    engine = InferenceEngine()

    submission_data = []

    # 3. Process Each Fragment
    for fid in fragment_ids:
        print(f"Processing Fragment {fid}...")

        # Run Decoupled Z-Scanning Inference
        prob_map = engine.predict_fragment(str(fid), load_cached_data=load_cached_data)

        # Binarize
        binary_map = (prob_map > Config.MASK_THRESHOLD).astype(np.uint8)

        # Encode
        rle = rle_encoding(binary_map)
        submission_data.append({"Id": fid, "Predicted": rle})

    # 4. Save Submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
