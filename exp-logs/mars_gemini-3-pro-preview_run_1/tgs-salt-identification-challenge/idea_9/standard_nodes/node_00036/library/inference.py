import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.dataset import SaltDataset
from library.model import DeepResUNet
from library.utils import rle_encode, seed_everything


def predict_snapshot(model_path, test_loader, device):
    """
    Loads a specific model checkpoint and generates predictions for the test set
    using Test-Time Augmentation (Horizontal Flip).

    Args:
        model_path (str): Path to the model checkpoint (.pth file).
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities with shape (N, 1, 128, 128).
                    Returns None if the model path does not exist.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Checkpoint not found at {model_path}. Skipping.")
        return None

    print(f"Loading checkpoint: {model_path}")
    model = DeepResUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for images, _, depths, _ in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # 1. Forward pass (Original)
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # 2. Forward pass (TTA: Horizontal Flip)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped, depths)
            probs_flipped = torch.sigmoid(logits_flipped)

            # Flip predictions back to original orientation
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            # Average original and flipped predictions
            batch_avg_probs = (probs + probs_flipped_back) / 2.0

            all_probs.append(batch_avg_probs.cpu().numpy())

    # Concatenate all batches: (N, 1, 128, 128)
    return np.concatenate(all_probs, axis=0)


def generate_submission(
    snapshot_paths,
    work_dir="./working/idea_9",
    output_path="./submission/submission.csv",
    batch_size=32,
    device_name="cuda",
    load_cached_data=True,
):
    """
    Generates a submission file by ensemble averaging predictions from multiple model snapshots.
    Performs cropping, thresholding, and RLE encoding.

    Args:
        snapshot_paths (list): List of file paths to model checkpoints.
        work_dir (str): Directory to store/load cached dataset files.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        device_name (str): Device to use ('cuda' or 'cpu').
        load_cached_data (bool): Whether to use cached data in SaltDataset.
    """
    seed_everything(42)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Starting inference on device: {device}")

    # 1. Prepare Test Data
    # Ensure work_dir exists for caching
    os.makedirs(work_dir, exist_ok=True)

    test_dataset = SaltDataset(
        mode="test", work_dir=work_dir, load_cached_data=load_cached_data
    )

    # Shuffle must be False to maintain alignment with dataset IDs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Collect Predictions from Snapshots
    ensemble_preds = []
    valid_snapshots = 0

    for path in snapshot_paths:
        preds = predict_snapshot(path, test_loader, device)
        if preds is not None:
            ensemble_preds.append(preds)
            valid_snapshots += 1

    if valid_snapshots == 0:
        print("Error: No valid model snapshots loaded. Aborting submission generation.")
        return

    print(f"Averaging predictions from {valid_snapshots} models...")

    # 3. Ensemble Averaging
    # shape: (N, 1, 128, 128)
    avg_preds = np.mean(ensemble_preds, axis=0)

    # 4. Post-processing
    # The model outputs 128x128 (padded). We need to crop to 101x101.
    # Padding calculation from dataset.py:
    # target=128, original=101 -> total_pad=27
    # p_l = 27 // 2 = 13
    # p_t = 27 // 2 = 13
    # Crop region: [13 : 13+101, 13 : 13+101]
    start_idx = 13
    end_idx = 13 + 101

    submission_rows = []
    ids = test_dataset.ids  # IDs are in the same order as loader because shuffle=False

    print("Encoding masks...")
    for i in range(len(ids)):
        img_id = ids[i]

        # Extract single image prediction
        full_prob_map = avg_preds[i, 0]

        # Crop to original size
        cropped_prob_map = full_prob_map[start_idx:end_idx, start_idx:end_idx]

        # Thresholding
        binary_mask = (cropped_prob_map > 0.5).astype(np.uint8)

        # RLE Encoding
        rle = rle_encode(binary_mask)

        submission_rows.append({"id": img_id, "rle_mask": rle})

    # 5. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission generated successfully at {output_path}")
