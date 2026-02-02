import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.model import DepthConditionedLinkNet
from library.dataset import SaltDataset, get_transforms
from library.utils import rle_encode, set_seed


def predict_and_submit(
    model_path="./working/idea_2/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=32,
    num_workers=4,
    load_cached_data=True,
):
    """
    Performs inference on the test set using the trained model, applies TTA,
    and generates a submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
        load_cached_data (bool): Whether to use cached dataset files.
    """
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Inference Device: {device}")
    print(f"Loading model from: {model_path}")

    # 2. Load Model
    # Initialize model structure
    model = DepthConditionedLinkNet(num_classes=1)

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Load Data
    print("Loading test data...")
    test_dataset = SaltDataset(
        mode="test",
        metadata_path="./metadata/test.csv",
        load_cached_data=load_cached_data,
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Inference Loop
    print(
        "Starting inference with TTA (Horizontal Flip) and Deep Supervision Ensemble..."
    )

    ids_list = []
    rle_list = []

    # Crop parameters for 128x128 -> 101x101
    # Padding was: Top=13, Bottom=14, Left=13, Right=14
    crop_top = 13
    crop_bottom = 13 + 101
    crop_left = 13
    crop_right = 13 + 101

    with torch.no_grad():
        for images, depths, image_ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # --- TTA Step 1: Original Image ---
            outputs_orig = model(images, depths)
            preds_orig = torch.sigmoid(outputs_orig)

            # --- TTA Step 2: Flipped Image ---
            # Horizontal flip (dim 3 for NCHW)
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip, depths)
            preds_flip_raw = torch.sigmoid(outputs_flip)

            # Flip back to original orientation
            preds_flip = torch.flip(preds_flip_raw, dims=[3])

            # --- Average TTA ---
            preds_avg = (preds_orig + preds_flip) / 2.0

            # --- Post-Processing ---
            # Convert to numpy
            preds_np = preds_avg.cpu().numpy()  # (B, 1, 128, 128)

            # Iterate over batch
            for i in range(preds_np.shape[0]):
                pred_mask = preds_np[i, 0, :, :]  # (128, 128)

                # Crop back to 101x101
                pred_mask = pred_mask[crop_top:crop_bottom, crop_left:crop_right]

                # Threshold
                binary_mask = (pred_mask > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(binary_mask)

                ids_list.append(image_ids[i])
                rle_list.append(rle)

    # 5. Save Submission
    print(f"Generating submission file at {output_path}...")
    submission_df = pd.DataFrame({"id": ids_list, "rle_mask": rle_list})

    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
