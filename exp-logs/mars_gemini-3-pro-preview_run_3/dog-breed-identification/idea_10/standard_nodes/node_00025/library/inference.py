import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, load_checkpoint
from library.model import get_model
from library.dataset import get_class_mapping, get_transforms, DogDataset

# Initialize logger
logger = get_logger("inference")


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The model to use for inference.
        loader (DataLoader): The data loader.
        device (torch.device): The device to run on.

    Returns:
        np.ndarray: Array of probabilities with shape (n_samples, n_classes).
        list: List of IDs corresponding to the predictions.
    """
    model.eval()
    all_probs = []
    all_ids = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass on horizontally flipped images
            # Images are (B, C, H, W). Dim 3 is Width.
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # 3. Average probabilities (TTA)
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def run_inference():
    """
    Main inference routine.
    Loads models from all folds, performs TTA prediction, ensembles results,
    and generates the submission file.
    """
    logger.info("Starting Inference Pipeline...")

    # 1. Setup
    device = Config.DEVICE

    # Get class mapping to ensure correct column order
    # We load cached data or compute if missing, handled by the utility function
    class_to_idx, class_names = get_class_mapping(load_cached_data=True)

    # 2. Data Preparation
    # Use 'val' transforms which are deterministic (Resize -> CenterCrop)
    test_transform = get_transforms("val")

    test_dataset = DogDataset(
        csv_path=Config.TEST_CSV,
        class_to_idx=class_to_idx,
        transform=test_transform,
        debug=Config.DEBUG,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"Test set size: {len(test_dataset)}")

    # 3. Ensemble Prediction
    n_folds = Config.N_FOLDS
    accumulated_probs = None
    sample_ids = None
    models_used = 0

    for fold in range(n_folds):
        model_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Loading model for Fold {fold}...")

        # Initialize model
        # We don't need to download pretrained weights since we are loading a full checkpoint
        model = get_model(device=device, pretrained=False)

        # Load checkpoint
        load_checkpoint(model_path, model, device=device)

        # Predict
        logger.info(f"Generating predictions for Fold {fold} with TTA...")
        probs, ids = predict_with_tta(model, test_loader, device)

        # Initialize accumulator if first successful fold
        if accumulated_probs is None:
            accumulated_probs = np.zeros_like(probs)
            sample_ids = ids
        else:
            # Verify ID alignment (sanity check)
            if ids != sample_ids:
                raise ValueError(f"ID mismatch in Fold {fold} predictions!")

        accumulated_probs += probs
        models_used += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError("No models were found for inference!")

    # 4. Average Predictions
    final_probs = accumulated_probs / models_used
    logger.info(f"Ensemble prediction complete using {models_used} models.")

    # 5. Generate Submission File
    logger.info("Generating submission file...")

    # Create DataFrame
    # Columns: id, breed1, breed2, ...
    df_sub = pd.DataFrame(final_probs, columns=class_names)
    df_sub.insert(0, "id", sample_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Inference finished successfully.")
