import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc

from library.config import (
    PATHS,
    INFERENCE_PARAMS,
    SPECIALIST_SETTINGS,
    MODEL_PARAMS,
    SLAB_PARAMS,
    DEVICE,
    SEED,
)
from library.model import get_model
from library.data_utils import get_fragment_3ch_slab


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    Pixels are numbered from left to right, then top to bottom (row-major).

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = mask.flatten()
    # Add a zero at the start and end to find transitions efficiently
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class InferenceEngine:
    def __init__(self):
        self.device = DEVICE
        self.models = {}
        self.tile_size = 512
        # Using non-overlapping tiles for efficiency as per baseline strategy
        self.stride = 512

        # Set seeds
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

    def load_models(self):
        """
        Loads the three specialist models into memory.
        """
        print("Loading specialist models...")
        for mode in SPECIALIST_SETTINGS.keys():
            model_path = os.path.join(PATHS.WORKING_DIR, f"model_{mode}.pth")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model checkpoint for {mode} not found at {model_path}. Skipping."
                )
                continue

            print(f"Loading {mode} model from {model_path}...")
            model = get_model(MODEL_PARAMS)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[mode] = model

        if not self.models:
            raise RuntimeError("No models were loaded. Cannot proceed with inference.")

    def predict_full_image(self, model, image):
        """
        Performs tiled inference on a large image.

        Args:
            model: PyTorch model.
            image: (H, W, 3) float32 numpy array, normalized [0,1].

        Returns:
            (H, W) float32 numpy array of probabilities.
        """
        h, w, c = image.shape

        # Pad image to be divisible by tile_size
        pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
        pad_w = (self.tile_size - w % self.tile_size) % self.tile_size

        image_padded = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
        h_padded, w_padded, _ = image_padded.shape

        prob_map = np.zeros((h_padded, w_padded), dtype=np.float32)

        # Prepare tiles
        tiles = []
        coords = []

        for y in range(0, h_padded, self.stride):
            for x in range(0, w_padded, self.stride):
                tile = image_padded[y : y + self.tile_size, x : x + self.tile_size, :]
                # HWC -> CHW
                tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1))
                tiles.append(tile_tensor)
                coords.append((y, x))

        # Batch processing
        batch_size = INFERENCE_PARAMS.get("batch_size", 8)

        with torch.no_grad():
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i : i + batch_size]
                batch_coords = coords[i : i + batch_size]

                # Stack batch
                input_tensor = torch.stack(batch_tiles).to(self.device)

                # Forward pass
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)

                # Place back
                probs_np = probs.cpu().numpy().squeeze(1)  # (B, H, W)

                for j, (y, x) in enumerate(batch_coords):
                    prob_map[y : y + self.tile_size, x : x + self.tile_size] = probs_np[
                        j
                    ]

        # Crop back to original size
        return prob_map[:h, :w]

    def process_fragment(self, fragment_id):
        """
        Processes a single fragment using the MDSE strategy.

        Args:
            fragment_id (str): Fragment ID.

        Returns:
            np.ndarray: Binary mask (H, W).
        """
        # 1. Load Validity Mask
        mask_path = os.path.join(PATHS.TEST_FRAGMENTS, fragment_id, "mask.png")
        if not os.path.exists(mask_path):
            # Fallback for training data testing
            mask_path = os.path.join(PATHS.TRAIN_FRAGMENTS, fragment_id, "mask.png")

        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if valid_mask is None:
            raise ValueError(f"Could not load mask for fragment {fragment_id}")

        valid_mask = (valid_mask > 0).astype(np.uint8)

        # 2. Specialist Predictions
        prob_maps = []

        for mode, settings in SPECIALIST_SETTINGS.items():
            if mode not in self.models:
                continue

            # Get specific view (High/Mid/Low)
            # This handles caching and projection
            print(f"  Generating view for Specialist {mode}...")
            image_slab = get_fragment_3ch_slab(
                fragment_id=fragment_id,
                split="test",
                z_start=settings["z_start"],
                z_end=settings["z_end"],
                slab_params=SLAB_PARAMS,
                load_cached_data=True,
            )

            # Predict
            print(f"  Running inference for Specialist {mode}...")
            prob = self.predict_full_image(self.models[mode], image_slab)
            prob_maps.append(prob)

            # Cleanup
            del image_slab
            gc.collect()

        if not prob_maps:
            return np.zeros_like(valid_mask)

        # 3. Max Fusion
        print("  Fusing predictions...")
        final_prob = prob_maps[0]
        for p in prob_maps[1:]:
            final_prob = np.maximum(final_prob, p)

        # 4. Masking and Thresholding
        threshold = INFERENCE_PARAMS.get("threshold", 0.5)
        binary_prediction = (final_prob > threshold).astype(np.uint8)

        # Apply validity mask
        binary_prediction = binary_prediction * valid_mask

        return binary_prediction

    def generate_submission(self):
        """
        Main entry point to generate submission.csv.
        """
        self.load_models()

        # Read Test Metadata
        if not os.path.exists(PATHS.TEST_METADATA):
            print("Test metadata not found. Cannot generate submission.")
            return

        df_test = pd.read_csv(PATHS.TEST_METADATA)
        ids = df_test["fragment_id"].unique()

        results = []

        print(f"Starting inference on {len(ids)} fragments...")

        for frag_id in ids:
            print(f"Processing Fragment {frag_id}...")
            frag_id_str = str(frag_id)

            try:
                binary_mask = self.process_fragment(frag_id_str)
                rle = rle_encode(binary_mask)
                results.append({"Id": frag_id_str, "Predicted": rle})
            except Exception as e:
                print(f"Error processing fragment {frag_id}: {e}")
                # Empty prediction on error
                results.append({"Id": frag_id_str, "Predicted": ""})

            gc.collect()
            torch.cuda.empty_cache()

        # Write Submission
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(PATHS.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {PATHS.SUBMISSION_FILE}")
