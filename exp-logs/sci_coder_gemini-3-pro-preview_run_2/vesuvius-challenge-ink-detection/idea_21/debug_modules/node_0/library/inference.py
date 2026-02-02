import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from library.config import Config
from library.model import SpecialistSegFormer
from library.data import project_volume_slab, load_fragment_slab
from library.utils import rle_encoding


def load_models(device):
    """
    Loads the three specialist models (A, B, C) from checkpoints.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        dict: A dictionary mapping specialist keys ('A', 'B', 'C') to loaded models.
    """
    models = {}
    specialists = ["A", "B", "C"]

    print(f"Loading specialist models on {device}...")

    for key in specialists:
        model_path = os.path.join(Config.WORKING_DIR, f"model_{key}_best.pth")

        # Initialize model architecture
        model = SpecialistSegFormer()

        if os.path.exists(model_path):
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded Specialist {key} from {model_path}")
        else:
            print(
                f"Warning: Checkpoint for Specialist {key} not found at {model_path}. "
                "Inference will proceed with initialized weights (random)."
            )

        model.to(device)
        model.eval()
        models[key] = model

    return models


def predict_fragment_ensemble(fragment_id, models, device):
    """
    Generates a binary segmentation mask for a full fragment using the ensemble.

    1. Loads the union of Z-slices required by all specialists.
    2. Generates 3 specific projections (A, B, C).
    3. Tiles the images, predicts with each specialist, and fuses via Max-Projection.

    Args:
        fragment_id (str): The ID of the fragment to predict.
        models (dict): Dictionary of loaded specialist models.
        device (torch.device): Compute device.

    Returns:
        np.ndarray: Binary mask (0 or 1) of the fragment.
    """
    # 1. Determine the Union Z-Range
    # A: 16-40, B: 20-44, C: 24-48 -> Union: 16-48
    union_start = 16
    union_end = 48

    # Load volume slab
    # Shape: (Depth=32, H, W)
    try:
        volume = load_fragment_slab(str(fragment_id), (union_start, union_end))
    except FileNotFoundError:
        print(f"Error: Volume data for fragment {fragment_id} not found.")
        return None

    depth, h, w = volume.shape

    # 2. Generate Projections
    # Relative indices within the loaded 32-slice volume
    # A: 16-40 -> relative 0-24
    # B: 20-44 -> relative 4-28
    # C: 24-48 -> relative 8-32

    projections = {}

    # Specialist A
    slab_a = volume[0:24]
    projections["A"] = project_volume_slab(slab_a)  # (H, W, 3)

    # Specialist B
    slab_b = volume[4:28]
    projections["B"] = project_volume_slab(slab_b)

    # Specialist C
    slab_c = volume[8:32]
    projections["C"] = project_volume_slab(slab_c)

    # Free volume memory
    del volume

    # 3. Tiling and Prediction
    tile_size = Config.TILE_SIZE

    # Calculate padding to make dimensions divisible by tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    # Pad projections
    padded_projections = {}
    for key, img in projections.items():
        # img is (H, W, 3)
        padded_projections[key] = np.pad(
            img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )

    h_padded, w_padded, _ = padded_projections["A"].shape

    # Initialize probability map
    # We will accumulate max probabilities here
    prob_map = np.zeros((h_padded, w_padded), dtype=np.float32)

    # Iterate over tiles
    # We process one tile at a time for all 3 models to fuse immediately and save memory
    # Or batch tiles. Given the complexity, processing row by row or batching is good.
    # Let's use a simple nested loop with batching if possible, but strict loop is safer for logic.

    batch_size = Config.BATCH_SIZE
    batch_coords = []

    # Create grid
    for y in range(0, h_padded, tile_size):
        for x in range(0, w_padded, tile_size):
            batch_coords.append((x, y))

    # Process in batches
    for i in range(0, len(batch_coords), batch_size):
        batch = batch_coords[i : i + batch_size]

        # Prepare inputs for each specialist
        # inputs[key] will be a tensor (B, 3, H, W)
        inputs = {k: [] for k in models.keys()}

        for x, y in batch:
            for key in models.keys():
                # Extract tile
                tile = padded_projections[key][y : y + tile_size, x : x + tile_size, :]

                # Normalize (uint8/uint16 -> float [0,1])
                # Assuming data is uint16 as per analysis (0-65535)
                # If it was uint8, this would be small, but analysis showed uint16.
                tile = tile.astype(np.float32) / 65535.0

                # Transpose to (3, H, W)
                tile = np.transpose(tile, (2, 0, 1))
                inputs[key].append(tile)

        # Convert to tensors
        tensor_inputs = {}
        for key in models.keys():
            tensor_inputs[key] = torch.tensor(np.array(inputs[key]), device=device)

        # Inference
        batch_preds = []  # List of tensors (B, 1, H, W)

        with torch.no_grad():
            for key, model in models.items():
                # Forward pass
                logits = model(tensor_inputs[key])
                probs = torch.sigmoid(logits)
                batch_preds.append(probs)

        # Stack and Max Fusion
        # Stack shape: (3, B, 1, H, W)
        stacked_preds = torch.stack(batch_preds, dim=0)
        # Max over specialists (dim 0)
        fused_preds, _ = torch.max(stacked_preds, dim=0)
        # fused_preds shape: (B, 1, H, W)

        # Place back into map
        fused_preds_np = fused_preds.cpu().numpy()

        for j, (x, y) in enumerate(batch):
            # Extract single prediction (1, H, W) -> (H, W)
            pred_tile = fused_preds_np[j, 0, :, :]
            prob_map[y : y + tile_size, x : x + tile_size] = pred_tile

    # Crop back to original size
    prob_map = prob_map[:h, :w]

    # Load Mask to mask out invalid areas
    # Paths are in metadata usually, but we can construct it
    mask_path = os.path.join(Config.INPUT_DIR, "test", str(fragment_id), "mask.png")
    # If not in test, check train (for demo/debugging purposes)
    if not os.path.exists(mask_path):
        mask_path = os.path.join(
            Config.INPUT_DIR, "train", str(fragment_id), "mask.png"
        )

    if os.path.exists(mask_path):
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if valid_mask is not None:
            # Resize if mismatch (should not happen usually)
            if valid_mask.shape != prob_map.shape:
                valid_mask = cv2.resize(
                    valid_mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
            prob_map = prob_map * (valid_mask > 0)

    # Threshold
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    return binary_mask


def create_submission():
    """
    Main inference pipeline.
    1. Loads models.
    2. Reads test metadata.
    3. Predicts each fragment.
    4. Encodes and writes submission.csv.
    """
    device = torch.device(Config.DEVICE)
    models = load_models(device)

    # Load test metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        print(f"Error: Test metadata not found at {test_csv_path}")
        return

    df_test = pd.read_csv(test_csv_path)
    fragment_ids = df_test["fragment_id"].unique()

    print(f"Found {len(fragment_ids)} test fragments.")

    submission_data = []

    for fid in fragment_ids:
        print(f"Processing fragment {fid}...")
        binary_mask = predict_fragment_ensemble(fid, models, device)

        if binary_mask is None:
            # Fallback for errors: empty prediction
            rle = ""
        else:
            rle = rle_encoding(binary_mask)

        submission_data.append({"Id": fid, "Predicted": rle})

    # Create DataFrame and save
    df_sub = pd.DataFrame(submission_data)

    # Ensure output path matches Config
    output_path = Config.SUBMISSION_PATH
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
