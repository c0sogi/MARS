import os
import glob
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import ContrailDataset
from library.model import ContrailUNet
from library.utils import rle_encode


def predict(load_cached_data=False):
    """
    Performs inference on the test set using an ensemble of the top checkpoints
    and Test Time Augmentation (TTA).

    Args:
        load_cached_data (bool): Flag to indicate if cached data should be loaded.
                                 (Not strictly used for final inference output generation
                                 as we want fresh predictions, but kept for signature consistency).
    """
    # ==============================
    # 1. Setup & Configuration
    # ==============================
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # Ensure submission directory exists
    submission_dir = os.path.dirname(Config.SUBMISSION_PATH)
    os.makedirs(submission_dir, exist_ok=True)

    # ==============================
    # 2. Identify Checkpoints
    # ==============================
    # We look for all checkpoint files saved by the training process.
    # train.py saves them as 'checkpoint_epoch_X_dice_Y.pth' in the checkpoint dir.
    checkpoint_pattern = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_*.pth")
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern))

    # Fallback logic: if no checkpoints in the specific folder, check for a 'best_model.pth'
    # which might be the averaged model from the end of training.
    if not checkpoint_paths:
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            print(
                f"No individual checkpoints found. Using averaged model: {best_model_path}"
            )
            checkpoint_paths = [best_model_path]
        else:
            raise FileNotFoundError(
                f"No valid checkpoints found in {Config.CHECKPOINT_DIR}"
            )
    else:
        print(f"Found {len(checkpoint_paths)} checkpoints for ensemble.")

    # ==============================
    # 3. Data Loading
    # ==============================
    print("Loading test dataset...")
    test_dataset = ContrailDataset(
        metadata_path=Config.TEST_METADATA_PATH, split="test"
    )

    # Use a slightly larger batch size for inference if memory allows,
    # but sticking to Config.BATCH_SIZE is safe.
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    num_samples = len(test_dataset)
    print(f"Total test samples to predict: {num_samples}")

    # ==============================
    # 4. Inference Loop (Ensemble + TTA)
    # ==============================
    # We will accumulate probabilities in a CPU tensor to avoid GPU OOM.
    # Shape: (N_samples, 1, H, W)
    accumulated_probs = torch.zeros(
        (num_samples, 1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
    )

    # Initialize model architecture
    model = ContrailUNet()
    model.to(device)

    for i, ckpt_path in enumerate(checkpoint_paths):
        print(
            f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {os.path.basename(ckpt_path)}"
        )

        # Load weights
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        batch_start_idx = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                batch_size = images.size(0)

                # --- TTA 1: Original Image ---
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # --- TTA 2: Horizontal Flip ---
                # Flip input along width (dim 3)
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip_flipped = torch.sigmoid(logits_flip)
                # Unflip output (flip back)
                probs_flip = torch.flip(probs_flip_flipped, dims=[3])

                # Average TTA views for this model
                batch_probs = (probs_orig + probs_flip) / 2.0

                # Accumulate to global CPU tensor
                batch_end_idx = batch_start_idx + batch_size
                accumulated_probs[batch_start_idx:batch_end_idx] += batch_probs.cpu()

                batch_start_idx = batch_end_idx

    # ==============================
    # 5. Averaging & Post-processing
    # ==============================
    print("Averaging ensemble predictions...")
    final_probs = accumulated_probs / len(checkpoint_paths)

    print("Generating submission file...")
    submission_rows = []
    record_ids = test_dataset.df["record_id"].values

    # Iterate through all samples to threshold and encode
    for idx in range(num_samples):
        # Extract probability map (H, W)
        prob_map = final_probs[idx, 0]

        # Threshold at 0.5
        binary_mask = (prob_map > 0.5).numpy().astype(np.uint8)

        # Run-Length Encoding
        rle_str = rle_encode(binary_mask)

        submission_rows.append(
            {"record_id": record_ids[idx], "encoded_pixels": rle_str}
        )

    # ==============================
    # 6. Save Submission
    # ==============================
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
