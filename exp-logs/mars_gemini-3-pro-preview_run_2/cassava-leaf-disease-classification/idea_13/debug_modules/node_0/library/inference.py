import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, load_checkpoint
from library.data import load_metadata, get_transforms, CassavaDataset
from library.model import CassavaModel

# Initialize logger
logger = get_logger()


def predict_test_set(debug: bool = False):
    """
    Performs inference on the test set using the trained models from all folds.
    Implements Ensemble Inference and Test Time Augmentation (TTA).

    Args:
        debug (bool): If True, runs inference on a small subset of data for testing.
    """
    device = Config.DEVICE
    logger.info(f"Starting inference on device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    # Load test metadata (handles debug slicing internally)
    df_test = load_metadata("test", debug=debug)
    logger.info(f"Test set size: {len(df_test)}")

    # Use Phase 2 image size (384) as the model was fine-tuned on this resolution
    transforms = get_transforms("test", img_size=Config.PHASE_2_IMG_SIZE)

    # Initialize Dataset and DataLoader
    # output_label=False because we might not have valid labels for test data
    test_dataset = CassavaDataset(df_test, transforms=transforms, output_label=False)

    # Use a slightly larger batch size for inference as no gradients are stored
    # Config.PHASE_2_BATCH_SIZE is 16; we can safely use 32.
    batch_size = Config.PHASE_2_BATCH_SIZE * 2

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Ensemble Inference Loop
    # -------------------------------------------------------------------------
    # Container for accumulated probabilities: Shape (Num_Samples, Num_Classes)
    final_probs = torch.zeros((len(df_test), Config.NUM_CLASSES), device=device)
    models_found = 0

    for fold in range(Config.NUM_FOLDS):
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        logger.info(f"Processing Fold {fold}...")

        # Initialize model architecture
        # pretrained=False because we are loading our own trained weights
        model = CassavaModel(
            model_name=Config.MODEL_NAME,
            pretrained=False,
            num_classes=Config.NUM_CLASSES,
        )
        model.to(device)
        model.eval()

        # Load weights
        try:
            load_checkpoint(checkpoint_path, model, device=device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint for fold {fold}: {e}")
            continue

        models_found += 1

        # Inference Loop for this Fold
        fold_probs = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # --- TTA Strategy ---

                # 1. Forward Pass: Original Image
                logits = model(images)
                probs = F.softmax(logits, dim=1)

                # 2. Forward Pass: Horizontal Flip (if enabled)
                if Config.TTA_FLIP:
                    # Flip along width dimension (dim 3 for NCHW tensor)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = F.softmax(logits_flipped, dim=1)

                    # Average original and flipped probabilities
                    probs = (probs + probs_flipped) / 2.0

                fold_probs.append(probs)

        # Concatenate batches for this fold and accumulate
        fold_probs = torch.cat(fold_probs, dim=0)
        final_probs += fold_probs

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    if models_found == 0:
        logger.error("No models were loaded. Cannot perform inference.")
        return

    # -------------------------------------------------------------------------
    # 3. Aggregation and Submission
    # -------------------------------------------------------------------------
    # Average probabilities across all folds
    avg_probs = final_probs / models_found

    # Get final predictions (Index of max probability)
    predictions = torch.argmax(avg_probs, dim=1).cpu().numpy()

    logger.info("Generating submission file...")

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"image_id": df_test["image_id"], "label": predictions}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Head of submission:\n{submission_df.head()}")
