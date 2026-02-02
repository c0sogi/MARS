import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import TumorDataset
from library.model import get_model


def run_inference(
    checkpoint_path=Config.CHECKPOINT_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Executes the inference pipeline with Test Time Augmentation (TTA).

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for the data loader.
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # Load test metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Loaded test metadata with {len(test_df)} samples.")

    # Initialize Dataset and DataLoader
    # split="test" ensures deterministic preprocessing (CenterCrop + Normalize)
    test_dataset = TumorDataset(test_df, split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to preserve ID order
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # 3. Model Loading
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = get_model()
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Checkpoint saved as dict with 'state_dict' key in train.py
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random/pretrained weights."
        )

    model.eval()

    # 4. Inference with TTA
    print("Running prediction with Test Time Augmentation (Steps: 4)...")
    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # TTA Step 1: Original
            # Output shape: (B, 1)
            out_orig = torch.sigmoid(model(images))

            # TTA Step 2: Horizontal Flip
            # Flip along width (dim 3 for NCHW)
            images_h = torch.flip(images, dims=[3])
            out_h = torch.sigmoid(model(images_h))

            # TTA Step 3: Vertical Flip
            # Flip along height (dim 2 for NCHW)
            images_v = torch.flip(images, dims=[2])
            out_v = torch.sigmoid(model(images_v))

            # TTA Step 4: Rotate 90
            # Rotate in the H-W plane (dims 2, 3)
            images_r = torch.rot90(images, k=1, dims=[2, 3])
            out_r = torch.sigmoid(model(images_r))

            # Average predictions
            avg_preds = (out_orig + out_h + out_v + out_r) / 4.0

            # Flatten and store
            all_preds.extend(avg_preds.flatten().cpu().tolist())

    # 5. Save Submission
    print(f"Generating submission file...")
    submission_df = pd.DataFrame({"id": test_df["id"], "label": all_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
