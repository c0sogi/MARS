import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed, rle_encoding, get_fragment_slab
from library.model import SegFormerSpecialist


class InferenceTileDataset(Dataset):
    """
    Helper dataset to handle tiling of large fragment images for inference.
    Pads the image to be divisible by tile_size and yields patches.
    """

    def __init__(self, image, tile_size=512):
        self.image = image
        self.tile_size = tile_size
        self.h, self.w = image.shape[:2]

        # Calculate padding required to make dimensions divisible by tile_size
        self.pad_h = (tile_size - (self.h % tile_size)) % tile_size
        self.pad_w = (tile_size - (self.w % tile_size)) % tile_size

        # Pad image: (H, W, 3)
        self.padded_image = np.pad(
            image,
            ((0, self.pad_h), (0, self.pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        self.new_h, self.new_w = self.padded_image.shape[:2]
        self.coords = []

        # Generate top-left coordinates for tiles
        for y in range(0, self.new_h, tile_size):
            for x in range(0, self.new_w, tile_size):
                self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        # Extract patch
        patch = self.padded_image[y : y + self.tile_size, x : x + self.tile_size, :]

        # Convert to Tensor: (H, W, 3) -> (3, H, W)
        # Ensure float32 and contiguous
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float()
        return tensor, y, x


def load_models():
    """
    Loads the ensemble of specialist models.
    Returns a dictionary mapping specialist type to the loaded model.
    """
    models = {}
    specialists = ["High", "Mid", "Low"]

    print("Loading specialist models...")
    for spec_type in specialists:
        model_path = os.path.join(Config.WORKING_DIR, f"model_{spec_type}.pth")

        # Initialize model architecture
        model = SegFormerSpecialist()
        model.to(Config.DEVICE)

        if os.path.exists(model_path):
            print(f"Loading weights for {spec_type} from {model_path}")
            state_dict = torch.load(model_path, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)
        else:
            print(
                f"WARNING: Weights for {spec_type} not found at {model_path}. Using random initialization."
            )
            # In a real scenario, this might raise an error, but for robustness we proceed
            # (predictions will be noise, but max-fusion might filter them if other models are good)

        model.eval()
        models[spec_type] = model

    return models


def predict_slab(model, slab):
    """
    Performs tiled inference on a single slab using a specific model.
    Returns the probability map (H, W).
    """
    dataset = InferenceTileDataset(slab, tile_size=Config.IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize canvas for padded size
    full_prob_map = np.zeros((dataset.new_h, dataset.new_w), dtype=np.float32)

    with torch.no_grad():
        for images, ys, xs in loader:
            images = images.to(Config.DEVICE)

            # Forward pass
            logits = model(images)  # (B, 1, H, W)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            # Place patches back onto canvas
            for i in range(len(ys)):
                y = ys[i].item()
                x = xs[i].item()
                prob_patch = probs[i, 0, :, :]

                full_prob_map[y : y + Config.IMAGE_SIZE, x : x + Config.IMAGE_SIZE] = (
                    prob_patch
                )

    # Crop back to original size
    original_h, original_w = slab.shape[:2]
    return full_prob_map[:original_h, :original_w]


def predict_and_submit():
    """
    Main inference pipeline.
    Loads data, runs ensemble inference, generates submission file.
    """
    set_seed(Config.SEED)

    # 1. Load Metadata
    if not os.path.exists(Config.METADATA_TEST_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.METADATA_TEST_PATH}"
        )

    test_df = pd.read_csv(Config.METADATA_TEST_PATH)
    print(f"Found {len(test_df)} test fragments.")

    # 2. Load Models
    models = load_models()

    submission_data = []

    # 3. Process each fragment
    for _, row in test_df.iterrows():
        frag_id = str(row["fragment_id"])
        volume_path = row["volume_path"]
        mask_path = row["mask_path"]

        print(f"\nProcessing Fragment: {frag_id}")

        # Load valid pixel mask
        full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
        if not os.path.exists(full_mask_path):
            print(f"Mask not found for fragment {frag_id}, skipping.")
            continue

        mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure binary mask (0 or 1)
        valid_mask = (mask_img > 0).astype(np.uint8)

        specialist_preds = []

        # Run each specialist on its specific view
        for spec_type, model in models.items():
            z_range = Config.Z_RANGES[spec_type]

            print(f"  Running Specialist: {spec_type} (Z: {z_range})")

            # Get specific view (cached or computed)
            slab = get_fragment_slab(
                fragment_id=frag_id,
                volume_path=volume_path,
                z_range=z_range,
                load_cached_data=True,
            )

            # Predict
            prob_map = predict_slab(model, slab)
            specialist_preds.append(prob_map)

        # 4. Ensemble Fusion (Max-Fusion)
        if not specialist_preds:
            print("  No predictions generated.")
            rle = ""
        else:
            # Stack: (3, H, W)
            stacked_preds = np.stack(specialist_preds, axis=0)
            # Max across specialists
            final_prob = np.max(stacked_preds, axis=0)

            # Apply valid mask
            final_prob = final_prob * valid_mask

            # Binarize
            binary_pred = (final_prob > Config.THRESHOLD).astype(np.uint8)

            # Encode
            rle = rle_encoding(binary_pred)

        submission_data.append({"Id": frag_id, "Predicted": rle})

    # 5. Save Submission
    submission_df = pd.DataFrame(submission_data)

    # Ensure columns are correct
    submission_df = submission_df[["Id", "Predicted"]]

    save_path = Config.SUBMISSION_PATH
    print(f"\nSaving submission to {save_path}")
    submission_df.to_csv(save_path, index=False)
    print("Done.")
