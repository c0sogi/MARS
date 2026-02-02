import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.model import HPUnet
from library.data import get_test_fragments


class TestPatchDataset(Dataset):
    """
    Dataset to yield tiles from a large fragment image for inference.
    """

    def __init__(self, image, tile_size, stride):
        self.image = image
        self.tile_size = tile_size
        self.stride = stride
        self.h, self.w = image.shape[:2]

        self.coords = []
        # Generate top-left coordinates for tiles
        # The image is assumed to be pre-padded to be divisible by tile_size
        for y in range(0, self.h, stride):
            for x in range(0, self.w, stride):
                # Ensure we don't go out of bounds (though padding should prevent this)
                if y + tile_size <= self.h and x + tile_size <= self.w:
                    self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        # Extract patch: (H, W, C)
        patch = self.image[y : y + self.tile_size, x : x + self.tile_size, :]
        # Convert to Tensor: (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        return patch, y, x


def predict_fragment(model, image, device):
    """
    Performs inference on a single fragment image using tiling and Test Time Augmentation (TTA).

    Args:
        model: Loaded PyTorch model.
        image: Input image array of shape (H, W, 4).
        device: Torch device.

    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    model.eval()
    h, w, c = image.shape
    tile_size = Config.TILE_SIZE
    stride = Config.STRIDE

    # Calculate padding to make dimensions divisible by tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    # Pad image with zeros (constant 0 is appropriate for background)
    # image is (H, W, C), so we pad first two dimensions
    image_padded = np.pad(
        image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
    )

    padded_h, padded_w = image_padded.shape[:2]

    # Create dataset and loader
    dataset = TestPatchDataset(image_padded, tile_size, stride)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize output map
    prob_map = torch.zeros((padded_h, padded_w), device=device, dtype=torch.float32)
    # Count map to handle overlaps (though stride=tile_size implies no overlap)
    count_map = torch.zeros((padded_h, padded_w), device=device, dtype=torch.float32)

    with torch.no_grad():
        for patches, ys, xs in loader:
            patches = patches.to(device)  # (B, C, H, W)

            # --- Test Time Augmentation (TTA) ---

            # 1. Original
            out = model(patches)
            preds = torch.sigmoid(out)

            # 2. Horizontal Flip
            patches_h = torch.flip(patches, dims=[3])
            out_h = model(patches_h)
            preds_h = torch.flip(torch.sigmoid(out_h), dims=[3])
            preds += preds_h

            # 3. Vertical Flip
            patches_v = torch.flip(patches, dims=[2])
            out_v = model(patches_v)
            preds_v = torch.flip(torch.sigmoid(out_v), dims=[2])
            preds += preds_v

            # 4. Rotate 90 (Counter-Clockwise)
            # Rotate spatial dims (2, 3)
            patches_r = torch.rot90(patches, k=1, dims=[2, 3])
            out_r = model(patches_r)
            # Rotate back (k=-1 or k=3)
            preds_r = torch.rot90(torch.sigmoid(out_r), k=-1, dims=[2, 3])
            preds += preds_r

            # Average predictions
            preds /= 4.0

            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            preds = preds.squeeze(1)

            # Place patches back into the full map
            for i in range(len(ys)):
                y, x = ys[i].item(), xs[i].item()
                prob_map[y : y + tile_size, x : x + tile_size] += preds[i]
                count_map[y : y + tile_size, x : x + tile_size] += 1.0

    # Normalize
    count_map[count_map == 0] = 1.0
    prob_map /= count_map

    # Crop back to original dimensions
    prob_map = prob_map[:h, :w]

    return prob_map.cpu().numpy()


def run_inference(load_cached_data=True):
    """
    Main inference pipeline.
    Loads data, loads model, predicts, and generates submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Test Metadata and Data
    # get_test_fragments handles loading and caching of 4-channel inputs
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    print("Loading test fragments...")
    fragments_data = get_test_fragments(test_df, load_cached_data=load_cached_data)

    # 2. Load Model
    model = HPUnet(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)

    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading model checkpoint from {Config.CHECKPOINT_PATH}")
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(
            f"WARNING: Checkpoint not found at {Config.CHECKPOINT_PATH}. Using random weights."
        )

    model.to(device)

    submission_data = []

    # 3. Inference Loop
    print(f"Starting inference on {len(fragments_data)} fragments...")

    for fid, data in fragments_data.items():
        image = data["image"]  # (H, W, 4)
        mask = data["mask"]  # (H, W) - Valid pixel mask

        # Predict
        prob_map = predict_fragment(model, image, device)

        # Thresholding
        binary_pred = (prob_map > Config.THRESHOLD).astype(np.uint8)

        # Apply Valid Mask (Force 0 outside the fragment)
        binary_pred = binary_pred * mask

        # RLE Encode
        rle_str = rle_encode(binary_pred)

        submission_data.append({"Id": fid, "Predicted": rle_str})

    # 4. Save Submission
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
