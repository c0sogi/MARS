import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import SegFormerB2
from library.utils import rle_encode


def predict_fragment(model, fragment_id, mask_shape, device, batch_size):
    """
    Generates a segmentation mask for a single fragment using Deterministic Z-Scanning
    and Max-Fusion strategy.

    Args:
        model (nn.Module): The loaded SegFormer model.
        fragment_id (str): The ID of the fragment to predict.
        mask_shape (tuple): The (Height, Width) of the fragment.
        device (torch.device): Compute device.
        batch_size (int): Batch size for inference.

    Returns:
        numpy.ndarray: The fused probability map (Height, Width).
    """
    h, w = mask_shape
    # Initialize the canvas for Max-Fusion (accumulate max probabilities)
    fused_map = np.zeros((h, w), dtype=np.float32)

    z_starts = Config.INFERENCE_Z_STARTS

    for z in z_starts:
        # Initialize dataset for this specific Z-depth
        dataset = InkDataset(mode="test", z_start=z)

        # Filter dataset to only include patches for the current fragment
        # This prevents processing all fragments for every Z-loop
        dataset.df = dataset.df[dataset.df["fragment_id"] == fragment_id].reset_index(
            drop=True
        )

        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                xs = batch["x"].numpy()
                ys = batch["y"].numpy()

                # Predict
                outputs = model(images)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

                # Place patches onto the canvas
                for i in range(len(images)):
                    prob = probs[i, 0, :, :]  # (512, 512)
                    x, y = xs[i], ys[i]

                    # Determine placement coordinates (handling boundaries)
                    y_end = min(y + Config.TILE_SIZE, h)
                    x_end = min(x + Config.TILE_SIZE, w)

                    h_copy = y_end - y
                    w_copy = x_end - x

                    if h_copy <= 0 or w_copy <= 0:
                        continue

                    # Crop prediction if it extends beyond image bounds (due to padding in dataset)
                    pred_region = prob[:h_copy, :w_copy]

                    # Max-Fusion: Update pixel values only if new prediction is higher
                    current_region = fused_map[y:y_end, x:x_end]
                    fused_map[y:y_end, x:x_end] = np.maximum(
                        current_region, pred_region
                    )

    return fused_map


def inference(checkpoint_path=None, batch_size=None):
    """
    Main inference routine.
    Generates predictions for all fragments in the test set and saves submission.csv.

    Args:
        checkpoint_path (str, optional): Path to model checkpoint. Defaults to best_model.pth.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
    """
    # Set defaults
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    device = torch.device(Config.DEVICE)
    print(f"Starting Inference on device: {device}")

    # 1. Load Model
    model = SegFormerB2()
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 2. Parse Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    # Get unique fragments
    unique_fragments = test_df[["fragment_id", "mask_path"]].drop_duplicates()

    submission_data = []

    print(f"Found {len(unique_fragments)} fragments to process.")

    # 3. Process Each Fragment
    for _, row in unique_fragments.iterrows():
        frag_id = str(row["fragment_id"])
        mask_rel_path = row["mask_path"]
        mask_full_path = os.path.join(Config.INPUT_DIR, mask_rel_path)

        # Load mask to get dimensions and for final masking
        valid_mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if valid_mask is None:
            print(f"Error: Could not load mask for fragment {frag_id}")
            continue

        h, w = valid_mask.shape
        print(f"Processing Fragment {frag_id} ({w}x{h})...")

        # Generate Probability Map
        prob_map = predict_fragment(model, frag_id, (h, w), device, batch_size)

        # 4. Post-processing
        # Binarize
        binary_pred = prob_map > Config.BINARIZATION_THRESHOLD

        # Apply Valid Mask (Ink only exists on the fragment surface)
        # valid_mask is 0 for background, >0 for fragment
        final_mask = np.logical_and(binary_pred, valid_mask > 0).astype(np.uint8)

        # 5. Encode
        rle_str = rle_encode(final_mask)
        submission_data.append({"Id": frag_id, "Predicted": rle_str})

    # 6. Save Submission
    sub_df = pd.DataFrame(submission_data)
    # Ensure correct column order
    sub_df = sub_df[["Id", "Predicted"]]
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
