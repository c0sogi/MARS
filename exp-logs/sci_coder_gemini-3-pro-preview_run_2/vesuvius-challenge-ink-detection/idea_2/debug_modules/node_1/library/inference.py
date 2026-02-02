import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode, create_submission
from library.model import InkDetector
from library.dataset import InkDataset


def predict_and_submit(load_cached_data=True):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed fragment volumes
                                 from disk to speed up initialization.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # 2. Load Metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata file not found at {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)

    # 3. Initialize Dataset and Loader
    # InkDataset in 'test' mode automatically generates tiles for the fragments
    dataset = InkDataset(test_df, mode="test", load_cached_data=load_cached_data)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Initialize Model
    model = InkDetector()
    model.to(Config.DEVICE)

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Best model not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    model.eval()

    # 5. Prepare Output Buffers
    # We need to reconstruct the full images from tiles.
    # We'll store the full probability map for each fragment.
    fragment_preds = {}
    fragment_masks = {}

    # Iterate over unique fragments in the original test_df to allocate memory
    for _, row in test_df.iterrows():
        frag_id = row["fragment_id"]
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

        # Load mask to get dimensions
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        fragment_preds[frag_id] = np.zeros(mask.shape, dtype=np.float32)
        fragment_masks[frag_id] = mask

    # 6. Inference Loop
    # We access the tiled dataframe inside the dataset to know where each tile belongs
    tiled_df = dataset.df
    current_idx = 0

    with torch.no_grad():
        for images in loader:
            batch_size = images.size(0)
            images = images.to(Config.DEVICE)

            # --- Test Time Augmentation (TTA) ---
            # 1. Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            if Config.USE_TTA:
                # 2. Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h)
                probs_h = torch.sigmoid(logits_h)
                probs += torch.flip(probs_h, dims=[3])

                # 3. Vertical Flip
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v)
                probs_v = torch.sigmoid(logits_v)
                probs += torch.flip(probs_v, dims=[2])

                # Average
                probs /= 3.0

            # Move to CPU
            probs = probs.cpu().numpy()  # (B, 1, H, W)

            # --- Reconstruction ---
            for b in range(batch_size):
                # Get tile metadata
                row = tiled_df.iloc[current_idx]
                frag_id = row["fragment_id"]
                x, y = int(row["x"]), int(row["y"])
                w, h = int(row["width"]), int(row["height"])

                # Extract prediction for this tile
                pred_tile = probs[b, 0, :, :]  # (H_tile, W_tile)

                # Determine valid area in the destination buffer
                # The dataset pads the input image if it goes out of bounds.
                # We must crop the prediction to the actual needed size.
                full_h, full_w = fragment_preds[frag_id].shape

                y_end = min(y + h, full_h)
                x_end = min(x + w, full_w)

                actual_h = y_end - y
                actual_w = x_end - x

                # Place prediction
                fragment_preds[frag_id][y:y_end, x:x_end] = pred_tile[
                    :actual_h, :actual_w
                ]

                current_idx += 1

    # 7. Post-processing and Encoding
    submission_dict = {}

    for frag_id, pred_map in fragment_preds.items():
        # Retrieve validity mask
        mask = fragment_masks[frag_id]
        valid_mask = (mask > 0).astype(bool)

        # Thresholding
        binary_pred = (pred_map > Config.THRESHOLD).astype(np.uint8)

        # Masking (remove predictions outside the papyrus fragment)
        binary_pred = binary_pred * valid_mask

        # RLE Encode
        rle = rle_encode(binary_pred)
        submission_dict[frag_id] = rle

    # 8. Save Submission
    create_submission(submission_dict, Config.SUBMISSION_PATH)
    print(f"Submission generated at {Config.SUBMISSION_PATH}")
