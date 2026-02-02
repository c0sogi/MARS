import os
import torch
import numpy as np
from library.config import ENSEMBLE_CONFIGS, DEVICE, NUM_WORKERS
from library.utils import get_logger, get_checkpoint_path, save_submission
from library.dataset import get_test_loader
from library.models import get_model
from library.engine import predict

# Initialize logger
logger = get_logger("inference")


def generate_ensemble_predictions(use_tta: bool = True):
    """
    Generates predictions for the test set using the heterogeneous ensemble.
    Iterates through all architectures and folds, applies TTA, averages the results,
    and saves the submission file.

    Args:
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal Flip) during inference.
                        Defaults to True.
    """
    logger.info("Starting ensemble inference...")

    final_probs = None
    final_ids = None
    model_count = 0

    # Iterate through each model architecture configuration
    for config in ENSEMBLE_CONFIGS:
        logger.info(
            f"Processing architecture: {config.model_name} (Size: {config.img_size})"
        )

        # Create test loader specific to the model's image size
        # Batch size can be slightly higher for inference than training, but we stick to config or safe default
        # Using config.batch_size * 2 for efficiency as no gradients are stored
        test_loader = get_test_loader(
            image_size=config.img_size,
            batch_size=config.batch_size * 2,
            num_workers=NUM_WORKERS,
        )

        # Iterate through each fold for the current architecture
        for fold in range(config.num_folds):
            checkpoint_path = get_checkpoint_path(config.name, fold)

            if not os.path.exists(checkpoint_path):
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}. Skipping this model."
                )
                continue

            logger.info(f"Loading model: {config.model_name}, Fold: {fold}")

            # Initialize model and load weights
            model = get_model(config.model_name, pretrained=False, num_classes=1)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.to(DEVICE)

            # Generate predictions (engine.predict handles TTA)
            ids, probs = predict(model, test_loader, DEVICE, use_tta=use_tta)

            # Initialize accumulator if this is the first successful prediction
            if final_probs is None:
                final_probs = np.zeros_like(probs)
                final_ids = ids

            # Verify ID alignment (sanity check)
            if not np.array_equal(final_ids, ids):
                raise ValueError(
                    f"ID mismatch detected for {config.model_name} fold {fold}"
                )

            # Accumulate probabilities
            final_probs += probs
            model_count += 1

            # Clean up to save memory
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        logger.error("No models were successfully loaded. Cannot generate submission.")
        return

    # Compute Arithmetic Mean
    avg_probs = final_probs / model_count
    logger.info(f"Ensemble prediction complete. Averaged over {model_count} models.")

    # Save Submission
    save_submission(final_ids, avg_probs)
    logger.info("Submission file saved successfully.")
