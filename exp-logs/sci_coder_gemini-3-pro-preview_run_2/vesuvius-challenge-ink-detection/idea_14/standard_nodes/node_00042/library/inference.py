import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encoding, load_checkpoint
from library.dataset import InkDataset
from library.model import SiameseSegFormer


class InferenceEngine:
    """
    Engine for running inference on test data using the Siamese Multi-View SegFormer.
    Handles TTA, tile stitching, and submission generation.
    """

    def __init__(self, checkpoint_path=Config.CHECKPOINT_PATH):
        """
        Args:
            checkpoint_path (str): Path to the model checkpoint.
        """
        self.checkpoint_path = checkpoint_path
        self.device = Config.DEVICE

        # Setup environment
        Config.setup()
        set_seed(Config.SEED)

        # Initialize Model
        print(f"Initializing model with backbone: {Config.MODEL_BACKBONE}")
        self.model = SiameseSegFormer(
            num_classes=Config.NUM_CLASSES,
            pretrained=False,  # No need to download pretrained weights, we load checkpoint
        )
        self.model.to(self.device)

        # Load Weights
        print(f"Loading checkpoint from {self.checkpoint_path}")
        score, epoch = load_checkpoint(
            self.model, path=self.checkpoint_path, device=self.device
        )
        print(f"Loaded model from epoch {epoch} with validation score {score}")

        self.model.eval()

    def _predict_batch_with_tta(self, views):
        """
        Performs Test Time Augmentation (TTA) on a batch of views.
        Augmentations: Identity, H-Flip, V-Flip, Rotate90.

        Args:
            views (dict): Dictionary containing 'view_1', 'view_2', 'view_3' tensors.

        Returns:
            torch.Tensor: Averaged probability map.
        """
        # Unpack views
        v1 = views["view_1"].to(self.device)
        v2 = views["view_2"].to(self.device)
        v3 = views["view_3"].to(self.device)

        probs_accum = None

        # Define TTA transformations
        # Format: (transform_func, inverse_func)
        transforms = [
            # Identity
            (lambda x: x, lambda x: x),
            # Horizontal Flip (dim 3 is width)
            (lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [3])),
            # Vertical Flip (dim 2 is height)
            (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),
            # Rotate 90 deg (k=1, dims=[2, 3])
            (lambda x: torch.rot90(x, 1, [2, 3]), lambda x: torch.rot90(x, -1, [2, 3])),
        ]

        with torch.no_grad():
            for t_func, inv_func in transforms:
                # Apply transform
                t_v1 = t_func(v1)
                t_v2 = t_func(v2)
                t_v3 = t_func(v3)

                # Forward pass
                logits = self.model(t_v1, t_v2, t_v3)
                probs = torch.sigmoid(logits)

                # Inverse transform
                inv_probs = inv_func(probs)

                if probs_accum is None:
                    probs_accum = inv_probs
                else:
                    probs_accum += inv_probs

        # Average
        probs_accum /= len(transforms)
        return probs_accum

    def generate_submission(self):
        """
        Main method to process all test fragments and generate submission.csv.
        """
        print("Starting Inference...")

        # Load Test Metadata
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA_PATH}"
            )

        test_df_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        submission_data = []

        # Process each fragment individually to manage memory
        unique_fragments = test_df_meta["fragment_id"].unique()

        for frag_id in unique_fragments:
            print(f"Processing Fragment {frag_id}...")

            # Filter metadata for this fragment
            frag_meta = test_df_meta[test_df_meta["fragment_id"] == frag_id].copy()

            # Initialize Dataset for this fragment
            # This handles tiling logic via _expand_test_dataframe inside InkDataset
            dataset = InkDataset(frag_meta, mode="test", load_cached_data=True)

            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,  # Critical for matching tiles to coordinates
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )

            # Determine canvas size from mask
            mask_path = os.path.join(Config.INPUT_DIR, frag_meta.iloc[0]["mask_path"])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"Could not load mask for fragment {frag_id}")

            h_img, w_img = mask_img.shape

            # Reconstruct probability map
            full_prob_map = np.zeros((h_img, w_img), dtype=np.float32)
            # Count map to handle overlaps if stride < tile_size (though current config is non-overlapping)
            count_map = np.zeros((h_img, w_img), dtype=np.float32)

            # Iterate over tiles
            # We need to access the dataset's expanded dataframe to get coordinates
            # Since shuffle=False, loader order matches dataset.df order

            batch_start_idx = 0

            for batch_idx, (inputs, _) in enumerate(loader):
                # Predict
                batch_probs = self._predict_batch_with_tta(inputs)  # (B, 1, H, W)
                batch_probs = batch_probs.cpu().numpy()

                batch_size = batch_probs.shape[0]

                # Place tiles
                for i in range(batch_size):
                    # Get metadata for this specific tile
                    tile_info = dataset.df.iloc[batch_start_idx + i]
                    x, y = int(tile_info["x"]), int(tile_info["y"])
                    w, h = int(tile_info["width"]), int(tile_info["height"])

                    # Extract prediction (remove channel dim)
                    pred_tile = batch_probs[i, 0, :, :]

                    # Handle boundaries (crop if prediction extends beyond image)
                    # The dataset logic pads inputs, but we only want the valid region
                    # InkDataset pads if (y+h) > frag_h.
                    # We need to crop the prediction to match the target placement area.

                    y_end = min(y + h, h_img)
                    x_end = min(x + w, w_img)

                    h_valid = y_end - y
                    w_valid = x_end - x

                    # Crop the valid part of the prediction (top-left is valid data)
                    pred_valid = pred_tile[:h_valid, :w_valid]

                    full_prob_map[y:y_end, x:x_end] += pred_valid
                    count_map[y:y_end, x:x_end] += 1.0

                batch_start_idx += batch_size

            # Normalize by count (averaging overlaps)
            # Avoid division by zero
            count_map[count_map == 0] = 1.0
            full_prob_map /= count_map

            # Apply Valid Mask (ignore background)
            valid_mask = (mask_img > 0).astype(np.float32)
            full_prob_map *= valid_mask

            # Threshold
            binary_mask = (full_prob_map > 0.5).astype(np.uint8)

            # RLE Encode
            rle = rle_encoding(binary_mask)

            submission_data.append({"Id": frag_id, "Predicted": rle})

            # Cleanup
            del full_prob_map, count_map, binary_mask, loader, dataset
            import gc

            gc.collect()

        # Create Submission DataFrame
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
