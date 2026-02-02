import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_loaders
from library.model import get_model


def load_ensemble_models(
    config: Config, device: torch.device, logger
) -> List[torch.nn.Module]:
    """
    Loads the trained models for all folds.

    Args:
        config (Config): Configuration object.
        device (torch.device): Device to load models onto.
        logger: Logger instance.

    Returns:
        List[torch.nn.Module]: List of loaded models in eval mode.
    """
    models = []

    for fold_idx in range(config.n_folds):
        model_path = os.path.join(config.working_dir, f"fold_{fold_idx}_best.pth")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping this fold.")
            continue

        logger.info(f"Loading model for fold {fold_idx} from {model_path}")

        # Instantiate model architecture
        model = get_model(config)
        model.to(device)

        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle case where checkpoint is a dict containing state_dict or just the state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load weights for fold {fold_idx}: {e}")
            continue

        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No models were loaded. Cannot proceed with inference.")

    return models


def predict_fn(config: Config = None):
    """
    Main inference function.

    1. Sets up environment.
    2. Loads test data (Phase 2 resolution).
    3. Loads ensemble of models.
    4. Performs inference with TTA (Horizontal Flip).
    5. Saves submission file.

    Args:
        config (Config, optional): Configuration object. If None, initializes a new one.
    """
    if config is None:
        config = Config()

    # Setup
    seed_everything(config.seed)
    logger = get_logger(os.path.join(config.working_dir, "inference.log"))
    device = config.device

    logger.info("Starting Inference...")
    logger.info(f"Configuration: {config}")

    # Data Loading
    # Use phase 2 for higher resolution (384x384) as used in fine-tuning
    logger.info("Loading test data (Phase 2 resolution)...")
    _, _, test_loader = get_loaders(config, phase=2, fold_idx=None)

    # Model Loading
    logger.info("Loading ensemble models...")
    models = load_ensemble_models(config, device, logger)
    logger.info(f"Successfully loaded {len(models)} models.")

    # Inference Loop
    image_ids = []
    final_preds = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Inference")):
            images = images.to(device)
            batch_size = images.size(0)

            # Placeholder for accumulated probabilities
            # Shape: (Batch_Size, Num_Classes)
            avg_probs = torch.zeros((batch_size, config.num_classes), device=device)

            # Test Time Augmentation (TTA)
            # 1. Original Images
            # 2. Horizontally Flipped Images
            images_flipped = torch.flip(images, dims=[3])  # [B, C, H, W], flip W

            inputs_list = [images]
            if config.tta:
                inputs_list.append(images_flipped)

            # Iterate over each model in the ensemble
            for model in models:
                for inp in inputs_list:
                    logits = model(inp)
                    probs = F.softmax(logits, dim=1)
                    avg_probs += probs

            # Normalize probabilities
            # Divide by (num_models * num_tta_views)
            divisor = len(models) * len(inputs_list)
            avg_probs /= divisor

            # Get final predictions (class index with max probability)
            preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

            # Store results
            # We need to retrieve image_ids. The dataset returns (image, label).
            # The test loader's dataset has the underlying dataframe.
            # We can calculate indices based on batch_idx and batch_size.
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + batch_size

            # Get image_ids from the dataset dataframe
            batch_image_ids = test_loader.dataset.df.iloc[start_idx:end_idx][
                "image_id"
            ].values

            image_ids.extend(batch_image_ids)
            final_preds.extend(preds)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "label": final_preds})

    # Save Submission
    logger.info(f"Saving submission to {config.submission_path}")
    submission_df.to_csv(config.submission_path, index=False)

    logger.info("Inference completed successfully.")

    # Print first few rows for verification
    print(submission_df.head())
