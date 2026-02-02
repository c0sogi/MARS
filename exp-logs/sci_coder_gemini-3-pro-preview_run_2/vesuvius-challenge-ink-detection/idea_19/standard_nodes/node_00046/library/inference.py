import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import SegFormerMiTB2
from library.utils import (
    generate_multiview_tensor,
    rle_encoding,
    write_submission,
    set_seed,
)


class InferenceEngine:
    """
    Engine for generating predictions using the Translation-Invariant SegFormer.
    Implements Multi-View Ensemble Scanning and Test Time Augmentation (TTA).
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = SegFormerMiTB2()
        self.model.to(self.device)
        self.model.eval()

        # Load the best model weights
        weights_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
        if os.path.exists(weights_path):
            print(f"Loading model weights from {weights_path}")
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
        else:
            print(
                f"Warning: Model weights not found at {weights_path}. Using random initialization."
            )

    def _predict_tile_tta(self, input_tensor):
        """
        Performs inference on a single tile with Test Time Augmentation (TTA).
        Augmentations: Original, Horizontal Flip, Vertical Flip.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (3, H, W).

        Returns:
            np.ndarray: Averaged probability map of shape (1, H, W).
        """
        # 1. Create Batch of Augmented Inputs
        # Original
        img = input_tensor
        # H-Flip (dim 2 is Width)
        img_h = torch.flip(input_tensor, dims=[2])
        # V-Flip (dim 1 is Height)
        img_v = torch.flip(input_tensor, dims=[1])

        # Stack into batch: (3, 3, H, W)
        batch = torch.stack([img, img_h, img_v]).to(self.device)

        # 2. Inference
        with torch.no_grad():
            logits = self.model(batch)  # (3, 1, H, W)
            probs = torch.sigmoid(logits)

        # 3. Inverse Augmentation
        p_orig = probs[0]  # (1, H, W)
        p_h = torch.flip(probs[1], dims=[2])  # Flip back width
        p_v = torch.flip(probs[2], dims=[1])  # Flip back height

        # 4. Average predictions
        p_avg = (p_orig + p_h + p_v) / 3.0

        return p_avg.cpu().numpy()

    def process_fragment(self, fragment_metadata):
        """
        Generates the binary prediction mask for a full fragment using Multi-View Scanning.

        Args:
            fragment_metadata (pd.Series): Row from the test metadata dataframe.

        Returns:
            str: Run-Length Encoded prediction string.
        """
        frag_id = fragment_metadata["fragment_id"]
        volume_path = os.path.join(Config.INPUT_DIR, fragment_metadata["volume_path"])
        mask_path = os.path.join(Config.INPUT_DIR, fragment_metadata["mask_path"])

        print(f"Processing fragment {frag_id}...")

        # Load binary mask to get dimensions
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            print(f"Error: Could not load mask for fragment {frag_id}")
            return ""

        h_img, w_img = binary_mask.shape

        # Initialize accumulator for Max-Fusion (float32 for precision)
        final_prob_map = np.zeros((h_img, w_img), dtype=np.float32)

        # Configuration for tiling
        tile_size = Config.TILE_SIZE
        stride = Config.TILE_SIZE  # Non-overlapping for efficiency

        # Iterate over the image in tiles
        for y in range(0, h_img, stride):
            for x in range(0, w_img, stride):

                # Calculate valid dimensions for this tile (handle edges)
                y_end = min(y + tile_size, h_img)
                x_end = min(x + tile_size, w_img)
                actual_h = y_end - y
                actual_w = x_end - x

                # Accumulator for this specific tile across views
                tile_max_prob = np.zeros((1, tile_size, tile_size), dtype=np.float32)

                # Multi-View Scanning: Iterate through A, B, C
                for view_name, start_z in Config.TRAIN_VIEWS.items():
                    # Generate input tensor for this view
                    # Function handles padding if crop is out of bounds
                    tensor = generate_multiview_tensor(
                        volume_path, start_z, x, y, tile_size, tile_size
                    )

                    # Predict with TTA
                    prob_view = self._predict_tile_tta(tensor)

                    # Update Max-Fusion accumulator
                    tile_max_prob = np.maximum(tile_max_prob, prob_view)

                # Crop the result to the actual image area (remove padding)
                # The model outputs 512x512; valid data is in the top-left
                valid_pred = tile_max_prob[0, :actual_h, :actual_w]

                # Place prediction into the full map
                final_prob_map[y:y_end, x:x_end] = valid_pred

        # Apply the fragment mask to zero out invalid background pixels
        final_prob_map = final_prob_map * (binary_mask > 0).astype(np.float32)

        # Threshold to binary
        binary_prediction = (final_prob_map > Config.THRESHOLD).astype(np.uint8)

        # Encode
        return rle_encoding(binary_prediction)

    def run(self, limit=None):
        """
        Main execution method to generate the submission file.

        Args:
            limit (int, optional): Limit the number of fragments processed (for debugging).
        """
        set_seed(Config.SEED)

        if not os.path.exists(Config.METADATA_TEST):
            print(f"Error: Test metadata not found at {Config.METADATA_TEST}")
            return

        df_test = pd.read_csv(Config.METADATA_TEST)

        if limit:
            df_test = df_test.head(limit)

        ids = []
        rles = []

        for _, row in df_test.iterrows():
            rle = self.process_fragment(row)
            ids.append(row["fragment_id"])
            rles.append(rle)

        write_submission(ids, rles, Config.SUBMISSION_PATH)


def run_inference(limit=None):
    """
    Wrapper function to run the inference engine.
    """
    engine = InferenceEngine()
    engine.run(limit=limit)
