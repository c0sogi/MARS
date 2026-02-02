import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_dataloaders
from library.model import DeepResUNet


def generate_submission(debug=Config.DEBUG):
    """
    Generates the submission file by running inference on the test set.

    Args:
        debug (bool): If True, runs on a subset of the data for debugging purposes.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Initializing Inference...")

    # 2. Data Loading
    # We only need the test_loader.
    # get_dataloaders handles the caching and preprocessing.
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=debug
    )

    # Replicate ID loading logic to ensure alignment with DataLoader
    # The DataLoader loads data sequentially, so we need the IDs in the same order.
    df_test = pd.read_csv(Config.TEST_CSV)
    if debug:
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    test_ids = df_test["id"].values

    # 3. Model Loading
    # Initialize model structure matching the training configuration
    model = DeepResUNet(
        in_channels=2, out_channels=1, deep_supervision=Config.DEEP_SUPERVISION
    )

    # Load weights
    if os.path.exists(Config.CHECKPOINT_PATH):
        state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.CHECKPOINT_PATH}")
    else:
        print(
            f"Warning: Checkpoint not found at {Config.CHECKPOINT_PATH}. Predictions will be based on random initialization."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_masks = []

    # Calculate cropping indices
    # Model output is 128x128 (padded), we need to crop back to 101x101
    pad_total = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_start = pad_total // 2
    pad_end = pad_start + Config.ORIG_SIZE
    crop_slice = slice(pad_start, pad_end)

    print("Starting inference with Test-Time Augmentation (TTA)...")

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # --- Test-Time Augmentation (TTA) ---

            # 1. Forward pass on original images
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Forward pass on horizontally flipped images
            # Flip width dimension (dim 3)
            images_flipped = torch.flip(images, dims=[3])
            out_flipped = model(images_flipped)
            prob_flipped = torch.sigmoid(out_flipped)
            # Flip predictions back to align with original
            prob_flipped = torch.flip(prob_flipped, dims=[3])

            # 3. Average predictions
            avg_prob = (prob_orig + prob_flipped) / 2.0

            # --- Post-Processing ---

            # Crop to original size (101x101)
            # avg_prob shape: (B, 1, 128, 128) -> (B, 101, 101)
            preds_cropped = avg_prob[:, 0, crop_slice, crop_slice]

            # Thresholding
            preds_bin = (preds_cropped > 0.5).byte().cpu().numpy()

            # Collect results
            for i in range(len(preds_bin)):
                all_masks.append(preds_bin[i])

    # 5. Submission Generation
    print("Encoding masks and saving submission...")

    rle_list = []
    for mask in all_masks:
        rle_list.append(rle_encode(mask))

    # Verify length match
    if len(test_ids) != len(rle_list):
        print(
            f"Warning: Number of IDs ({len(test_ids)}) does not match number of predictions ({len(rle_list)})."
        )

    sub_df = pd.DataFrame({"id": test_ids, "rle_mask": rle_list})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
