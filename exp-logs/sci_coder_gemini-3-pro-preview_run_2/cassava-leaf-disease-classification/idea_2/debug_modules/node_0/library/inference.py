import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import (
    tqdm,
)  # Not used for printing, but good practice to have if needed, though forbidden by prompt logic "no progress bars"

from library.config import Config
from library.utils import get_logger, seed_everything
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaClassifier


def run_inference(subset_size=None):
    """
    Performs inference on the test dataset using Test Time Augmentation (TTA).
    Generates the submission.csv file.

    Args:
        subset_size (int, optional): Limit the number of test samples (for debugging).
    """
    # 1. Setup
    logger = get_logger(os.path.join(Config.WORKING_DIR, "inference.log"))
    logger.info("Starting inference process...")

    seed_everything(Config.SEED)
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Initializing Test Dataset and DataLoader...")
    test_dataset = CassavaDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        transform=get_transforms("test"),
        data_split="test",
        subset_size=subset_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = CassavaClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # No need to download pretrained weights, we load checkpoint
        num_classes=Config.NUM_CLASSES,
    )

    checkpoint_path = Config.MODEL_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Using random initialization (Expect poor results)."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop with TTA
    logger.info("Running inference with TTA (Original + Horizontal Flip)...")

    all_preds = []
    # We need to track image_ids to ensure alignment, though DataLoader preserves order
    # The dataset class provides access to image_ids via index, but simpler to collect sequentially
    # since shuffle=False.

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward Pass 1: Original Images
            logits_orig = model(images)

            # Forward Pass 2: Horizontally Flipped Images (TTA)
            # Images are [Batch, Channel, Height, Width]. Flip on dim 3 (Width).
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped)

            # Average Logits
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Get Predictions
            preds = torch.argmax(avg_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # 5. Generate Submission
    logger.info("Generating submission file...")

    # Retrieve image IDs from the dataset
    # If subset_size was used, the dataset handles slicing internally
    image_ids = test_dataset.image_ids

    # Sanity check
    if len(image_ids) != len(all_preds):
        logger.error(f"Mismatch: {len(image_ids)} IDs vs {len(all_preds)} predictions.")

    submission_df = pd.DataFrame({"image_id": image_ids, "label": all_preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Head of submission:\n{submission_df.head()}")

    return submission_df
