import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.model import SegFormerB3
from library.data_processing import load_fragment_slab
from library.utils import rle_encoding


class InferenceTileDataset(Dataset):
    """
    Dataset to yield tiles from a large fragment image for inference.
    Handles padding to ensure full coverage by the fixed tile size.
    """

    def __init__(self, image: np.ndarray, tile_size: int = 512):
        self.image = image  # Shape: (H, W, C)
        self.tile_size = tile_size
        self.h, self.w = image.shape[:2]

        # Calculate required padding to make dimensions divisible by tile_size
        self.pad_h = (tile_size - (self.h % tile_size)) % tile_size
        self.pad_w = (tile_size - (self.w % tile_size)) % tile_size

        # Pad the image (bottom and right)
        # ((top, bottom), (left, right), (channels_before, channels_after))
        self.padded_image = np.pad(
            image,
            ((0, self.pad_h), (0, self.pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        self.new_h, self.new_w = self.padded_image.shape[:2]

        # Generate tile coordinates
        self.coords = []
        for y in range(0, self.new_h, tile_size):
            for x in range(0, self.new_w, tile_size):
                self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        # Extract tile
        tile = self.padded_image[y : y + self.tile_size, x : x + self.tile_size, :]

        # Convert to Tensor (C, H, W)
        # Normalize is already done in load_fragment_slab (returns float32 [0,1])
        tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float()

        return tile_tensor, y, x


def predict_fragment_scan(
    model: torch.nn.Module, fragment_id: str, z_start: int, device: str
) -> np.ndarray:
    """
    Generates a probability map for a specific fragment at a specific Z-depth.
    Uses Test Time Augmentation (TTA) and tiling.
    """
    # 1. Resolve Volume Path from Metadata
    # We read the test metadata to find where the volume is located
    df_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    # Ensure fragment_id is string for comparison
    row_df = df_test[df_test["fragment_id"].astype(str) == str(fragment_id)]

    if row_df.empty:
        raise ValueError(f"Fragment {fragment_id} not found in test metadata.")

    volume_path = row_df.iloc[0]["volume_path"]

    # 2. Load Input Slab
    # load_fragment_slab handles caching and 3D->2D projection
    slab = load_fragment_slab(
        str(fragment_id), volume_path, z_start, load_cached_data=True
    )

    # 3. Prepare Tiled Inference
    dataset = InferenceTileDataset(slab, tile_size=Config.TILE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # Container for predictions (padded size)
    probs_map = torch.zeros((dataset.new_h, dataset.new_w), dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        for tiles, ys, xs in loader:
            tiles = tiles.to(device)  # (B, 3, H, W)

            # --- Test Time Augmentation (TTA) ---
            # We average predictions from: Original, Horizontal Flip, Vertical Flip

            # Pass 1: Original
            logits_1 = model(tiles)
            preds_1 = torch.sigmoid(logits_1)

            # Pass 2: Horizontal Flip
            tiles_h = torch.flip(tiles, dims=[3])
            logits_h = model(tiles_h)
            preds_h = torch.sigmoid(logits_h)
            preds_h = torch.flip(preds_h, dims=[3])  # Flip back

            # Pass 3: Vertical Flip
            tiles_v = torch.flip(tiles, dims=[2])
            logits_v = model(tiles_v)
            preds_v = torch.sigmoid(logits_v)
            preds_v = torch.flip(preds_v, dims=[2])  # Flip back

            # Average predictions
            avg_preds = (preds_1 + preds_h + preds_v) / 3.0

            # --- Reassemble ---
            # avg_preds is (B, 1, H, W)
            avg_preds = avg_preds.squeeze(1).cpu()

            for i in range(tiles.size(0)):
                y = ys[i].item()
                x = xs[i].item()
                h_t = Config.TILE_SIZE
                w_t = Config.TILE_SIZE

                # Place tile in the map
                probs_map[y : y + h_t, x : x + w_t] = avg_preds[i]

    # 4. Crop to original dimensions
    # The dataset class stored the original h, w
    final_probs = probs_map[: dataset.h, : dataset.w].numpy()

    return final_probs


def z_scan_inference():
    """
    Main inference routine implementing Decoupled Z-Scanning.
    Orchestrates multi-pass prediction, max-fusion, and RLE submission generation.
    """
    device = Config.DEVICE
    print(f"Inference Device: {device}")

    # 1. Load Model
    model = SegFormerB3()
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Train the model first."
        )

    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load Test Metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    submission_results = []

    print(f"Found {len(df_test)} fragments for inference.")

    # 3. Process each fragment
    for idx, row in df_test.iterrows():
        frag_id = str(row["fragment_id"])
        mask_path = row["mask_path"]

        print(f"Processing Fragment {frag_id}...")

        # Load the binary mask to define valid regions and dimensions
        full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
        mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            print(
                f"Warning: Mask not found for fragment {frag_id} at {full_mask_path}. Skipping."
            )
            continue

        h, w = mask_img.shape
        valid_mask = mask_img > 0

        # Initialize accumulator for Max-Fusion
        fused_probs = np.zeros((h, w), dtype=np.float32)

        # 4. Multi-Pass Z-Scanning
        # Iterate through the defined Z-starts (e.g., 18, 20, 22)
        for z_start in Config.INFERENCE_Z_STARTS:
            print(f"  Scanning depth Z={z_start}...")

            try:
                # Generate probability map for this depth
                scan_probs = predict_fragment_scan(model, frag_id, z_start, device)

                # Update fused map with maximum probability per pixel
                fused_probs = np.maximum(fused_probs, scan_probs)

            except Exception as e:
                print(f"  Error processing Z={z_start} for fragment {frag_id}: {e}")
                # Continue to next scan/fragment rather than crashing entirely
                continue

        # 5. Post-Processing
        # Mask out invalid regions (outside the fragment)
        fused_probs[~valid_mask] = 0

        # Thresholding
        binary_prediction = (fused_probs > 0.5).astype(np.uint8)

        # Encode
        rle_str = rle_encoding(binary_prediction)
        submission_results.append({"Id": frag_id, "Predicted": rle_str})

    # 6. Save Submission
    submission_df = pd.DataFrame(submission_results)

    # Ensure columns are in correct order
    submission_df = submission_df[["Id", "Predicted"]]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
