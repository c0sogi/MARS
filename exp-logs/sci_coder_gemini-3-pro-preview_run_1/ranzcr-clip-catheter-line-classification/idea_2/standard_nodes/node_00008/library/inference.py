import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import get_device, seed_everything
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel


def predict(debug: bool = False, debug_size: int = 100):
    """
    Performs inference on the test set using the trained model with TTA.
    Generates the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of data.
        debug_size (int): Number of samples to use in debug mode.
    """
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Starting inference on device: {device}")

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    test_df = pd.read_csv(Config.TEST_METADATA)
    print(f"Loaded test metadata with {len(test_df)} samples.")

    # 2. Initialize Dataset and DataLoader
    # get_transforms("test") provides Resize and Normalize, but NOT augmentation/flipping.
    test_dataset = CatheterDataset(
        test_df,
        transforms=get_transforms("test"),
        mode="test",
        debug=debug,
        debug_size=debug_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    # We set pretrained=False because we are loading our own trained weights.
    model = CatheterModel(model_name=Config.MODEL_NAME, pretrained=False)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(
            f"Warning: Model weights not found at {model_path}. Using random initialization (for debugging only)."
        )
    else:
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop with TTA (Original + Horizontal Flip)
    all_preds = []
    print("Running inference with TTA (Original + Horizontal Flip)...")

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True)

            # --- TTA Strategy ---

            # Pass 1: Original Images
            with autocast(enabled=Config.USE_AMP):
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

            # Pass 2: Horizontally Flipped Images
            # Tensor shape is (B, C, H, W). Horizontal flip is along dim 3 (W).
            images_flipped = torch.flip(images, dims=[3])

            with autocast(enabled=Config.USE_AMP):
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

            # Average the probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_preds.append(probs_avg.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_preds, axis=0)

    # 5. Generate Submission File
    # If debug mode was used, slice the dataframe to match the number of predictions
    if debug:
        submission_df = test_df.iloc[:debug_size].copy()
    else:
        submission_df = test_df.copy()

    # Keep only the identifier column
    submission_df = submission_df[["StudyInstanceUID"]].reset_index(drop=True)

    # Assign predictions to target columns in the correct order
    for i, col in enumerate(Config.TARGET_COLS):
        submission_df[col] = predictions[:, i]

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
