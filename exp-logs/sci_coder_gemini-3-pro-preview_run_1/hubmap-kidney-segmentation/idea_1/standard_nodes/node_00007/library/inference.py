import os
import gc
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import rle_encode
from library.model import LinkNetResNet18


class InferenceTileDataset(Dataset):
    """
    Dataset that yields tiles from a single large image for sliding window inference.
    """

    def __init__(self, image_path, tile_size, overlap):
        self.image_path = image_path
        self.tile_size = tile_size

        # Open image briefly to get dimensions
        with rasterio.open(image_path) as src:
            self.H, self.W = src.height, src.width

        # Generate grid coordinates
        # Stride is the step size. overlap=0.25 means stride is 0.75 * tile_size
        stride = int(tile_size * (1 - overlap))

        self.coordinates = []

        # Calculate Y coordinates
        y_points = list(range(0, self.H - tile_size + 1, stride))
        # Ensure the bottom edge is covered
        if (self.H - tile_size) % stride != 0:
            y_points.append(max(0, self.H - tile_size))
        # Handle images smaller than tile_size
        if self.H < tile_size:
            y_points = [0]

        # Calculate X coordinates
        x_points = list(range(0, self.W - tile_size + 1, stride))
        # Ensure the right edge is covered
        if (self.W - tile_size) % stride != 0:
            x_points.append(max(0, self.W - tile_size))
        # Handle images smaller than tile_size
        if self.W < tile_size:
            x_points = [0]

        for y in y_points:
            for x in x_points:
                self.coordinates.append((x, y))

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        x, y = self.coordinates[idx]

        # Read specific window using rasterio
        # boundless=True automatically pads with fill_value if window extends beyond image
        with rasterio.open(self.image_path) as src:
            window = Window(x, y, self.tile_size, self.tile_size)
            if src.count == 3:
                img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            else:
                img = src.read([1], window=window, boundless=True, fill_value=0)
                img = np.repeat(img, 3, axis=0)

            # Convert (C, H, W) -> (H, W, C) for consistency if needed,
            # but here we just need to normalize and return to (C, H, W) for PyTorch
            img = np.moveaxis(img, 0, -1)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), x, y


def predict_sliding_window(model, image_path, device):
    """
    Performs sliding window inference on a large image.

    Args:
        model: Trained PyTorch model.
        image_path: Path to the TIFF image.
        device: Torch device.

    Returns:
        np.ndarray: Binary mask of the full image.
    """
    tile_size = Config.TILE_SIZE
    overlap = Config.INFERENCE_OVERLAP

    # Initialize dataset and loader
    dataset = InferenceTileDataset(image_path, tile_size, overlap)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Get image dimensions
    with rasterio.open(image_path) as src:
        H, W = src.height, src.width

    # Allocate memory for probability map and count map (for averaging)
    # Using float32 on CPU to accommodate large images (220GB RAM available)
    prob_map = torch.zeros((H, W), dtype=torch.float32, device="cpu")
    count_map = torch.zeros((H, W), dtype=torch.float32, device="cpu")

    model.eval()

    with torch.no_grad():
        for images, x_coords, y_coords in loader:
            images = images.to(device)

            # Forward pass with AMP
            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)

            # Move to CPU for accumulation
            # Ensure float32 for accumulation
            probs = probs.to("cpu").float()

            # Accumulate predictions
            for i in range(len(x_coords)):
                x = x_coords[i].item()
                y = y_coords[i].item()
                p = probs[i]

                # Determine valid region in the global map
                # The tile is always tile_size x tile_size (padded if needed)
                # We crop the prediction to fit the image bounds if we are at the edge
                h_end = min(y + tile_size, H)
                w_end = min(x + tile_size, W)

                h_len = h_end - y
                w_len = w_end - x

                # Add probabilities and increment counts
                prob_map[y:h_end, x:w_end] += p[:h_len, :w_len]
                count_map[y:h_end, x:w_end] += 1.0

    # Calculate average probability
    # Avoid division by zero (though logic ensures count >= 1)
    avg_prob = prob_map / (count_map + 1e-6)

    # Apply threshold
    mask = (avg_prob > Config.THRESHOLD).numpy().astype(np.uint8)

    # Clean up
    del prob_map, count_map, avg_prob
    gc.collect()

    return mask


def generate_submission(
    checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    output_path=Config.SUBMISSION_PATH,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Generates the submission file by running inference on the test set.
    """
    print("Starting submission generation...")

    # 1. Load Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(f"Debug mode: Processing first {debug_sample_size} images.")
        test_df = test_df.head(debug_sample_size)

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = LinkNetResNet18(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using random weights.")

    model.to(device)

    # 3. Inference Loop
    results = []

    for idx, row in test_df.iterrows():
        img_id = row["id"]
        img_path = row["image_path"]

        print(f"Processing image {idx+1}/{len(test_df)}: {img_id}")

        try:
            # Run inference
            mask = predict_sliding_window(model, img_path, device)

            # Encode mask
            rle = rle_encode(mask)

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            rle = ""

        results.append({"id": img_id, "predicted": rle})

        # Explicit garbage collection to free memory between large images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
