import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import rle_encoding
from library.model import StratifiedSegFormer
from library.data import load_or_generate_fragment_mips


class TestTileDataset(Dataset):
    """
    Dataset to yield tiles from a large fragment image for inference.
    """

    def __init__(self, image, tile_size=512):
        self.image = image
        self.tile_size = tile_size
        self.h, self.w = image.shape[:2]

        # Calculate padding to make image divisible by tile_size
        self.pad_h = (tile_size - (self.h % tile_size)) % tile_size
        self.pad_w = (tile_size - (self.w % tile_size)) % tile_size

        # Pad image
        self.padded_image = np.pad(
            image, ((0, self.pad_h), (0, self.pad_w), (0, 0)), mode="constant"
        )

        self.new_h, self.new_w = self.padded_image.shape[:2]
        self.n_rows = self.new_h // tile_size
        self.n_cols = self.new_w // tile_size

    def __len__(self):
        return self.n_rows * self.n_cols

    def __getitem__(self, idx):
        # Calculate row and col index
        row = idx // self.n_cols
        col = idx % self.n_cols

        y1 = row * self.tile_size
        y2 = y1 + self.tile_size
        x1 = col * self.tile_size
        x2 = x1 + self.tile_size

        patch = self.padded_image[y1:y2, x1:x2, :]

        # Normalize: uint16 -> float32 [0, 1]
        patch = patch.astype(np.float32) / 65535.0

        # To Tensor: (H, W, C) -> (C, H, W)
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)

        return patch_tensor, row, col


def predict_batch_tta(model, images, device):
    """
    Performs inference with Test Time Augmentation.
    """
    images = images.to(device)

    # 1. Original
    logits = model(images)
    probs = torch.sigmoid(logits)

    if Config.USE_TTA:
        # 2. Horizontal Flip
        images_hf = torch.flip(images, dims=[3])
        logits_hf = model(images_hf)
        probs_hf = torch.sigmoid(logits_hf)
        probs += torch.flip(probs_hf, dims=[3])

        # 3. Vertical Flip
        images_vf = torch.flip(images, dims=[2])
        logits_vf = model(images_vf)
        probs_vf = torch.sigmoid(logits_vf)
        probs += torch.flip(probs_vf, dims=[2])

        # 4. Rotate 90 (k=1, dims=(2,3))
        images_r90 = torch.rot90(images, k=1, dims=[2, 3])
        logits_r90 = model(images_r90)
        probs_r90 = torch.sigmoid(logits_r90)
        probs += torch.rot90(probs_r90, k=-1, dims=[2, 3])

        # Average
        probs /= 4.0

    return probs


def predict_fragment(model, fragment_id, volume_path, device, load_cached_data=True):
    """
    Generates a binary mask for a single fragment using sliding window inference.
    """
    # 1. Load MIPs (Cached or Generated)
    # The load_or_generate function handles the caching logic internally
    image = load_or_generate_fragment_mips(
        fragment_id, volume_path, load_cached_data=load_cached_data
    )

    original_h, original_w = image.shape[:2]

    # 2. Prepare Dataset and Loader
    dataset = TestTileDataset(image, tile_size=Config.TILE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Inference Loop
    # Canvas to store predictions
    prediction_map = torch.zeros(
        (dataset.new_h, dataset.new_w), dtype=torch.float32, device="cpu"
    )

    model.eval()
    with torch.no_grad():
        for images, rows, cols in loader:
            # Predict
            probs = predict_batch_tta(model, images, device)

            # Place on canvas
            probs = probs.squeeze(1).cpu()  # (B, H, W)

            for i in range(len(images)):
                r, c = rows[i].item(), cols[i].item()
                y1 = r * Config.TILE_SIZE
                y2 = y1 + Config.TILE_SIZE
                x1 = c * Config.TILE_SIZE
                x2 = x1 + Config.TILE_SIZE

                prediction_map[y1:y2, x1:x2] = probs[i]

    # 4. Post-processing
    # Crop padding
    prediction_map = prediction_map[:original_h, :original_w]

    # Threshold
    binary_mask = (prediction_map > Config.THRESHOLD).numpy().astype(np.uint8)

    return binary_mask


def generate_submission(load_cached_data=True):
    """
    Main inference routine. Generates submission.csv.
    """
    print("Starting Inference...")

    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Load Metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    print(f"Found {len(df_test)} test fragments.")

    # 2. Load Model
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train the model first."
        )

    model = StratifiedSegFormer(
        pretrained=False
    )  # Pretrained weights not needed for inference, we load state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    submission_data = []

    # 3. Process Fragments
    for _, row in df_test.iterrows():
        frag_id = str(row["fragment_id"])
        vol_path = row["volume_path"]

        print(f"Processing Fragment {frag_id}...")

        # Predict
        binary_mask = predict_fragment(
            model, frag_id, vol_path, device, load_cached_data=load_cached_data
        )

        # Encode
        rle = rle_encoding(binary_mask)
        submission_data.append({"Id": frag_id, "Predicted": rle})

    # 4. Save Submission
    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
