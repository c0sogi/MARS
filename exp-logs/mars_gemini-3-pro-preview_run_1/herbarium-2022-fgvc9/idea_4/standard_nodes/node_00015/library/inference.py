import os
import torch
import pandas as pd
import numpy as np
from library.utils import Config, get_device, ensure_dirs, setup_logger
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt


def predict_tta(model, loader, device):
    """
    Performs inference using Test-Time Augmentation (TTA).

    For each image, predictions are generated for both the original and a
    horizontally flipped version. The logits are averaged before applying argmax.

    Args:
        model (nn.Module): The trained HierarchicalConvNeXt model.
        loader (DataLoader): The test set DataLoader.
        device (torch.device): The computation device (CPU or CUDA).

    Returns:
        tuple: (predictions, image_ids)
            - predictions (list): List of predicted model indices (int).
            - image_ids (list): List of image identifiers (str).
    """
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # TTA: Create horizontally flipped version of the batch
            # Flip along the width dimension (dim 3 for NCHW format)
            images_flipped = torch.flip(images, dims=[3])

            # Forward pass for original images
            outputs_orig = model(images)
            logits_orig = outputs_orig["species"]

            # Forward pass for flipped images
            outputs_flip = model(images_flipped)
            logits_flip = outputs_flip["species"]

            # Average the logits to reduce variance
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Determine class index with highest probability
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    return predictions, image_ids


def generate_submission(model_path=None, output_file="submission.csv"):
    """
    Orchestrates the submission generation process.

    Loads the test data, initializes the model, loads weights, performs TTA inference,
    maps predictions back to original category IDs, and saves the CSV.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to Config.WORKING_DIR/best_model.pth.
        output_file (str, optional): Name of the output CSV file.
                                     Defaults to 'submission.csv'.
    """
    ensure_dirs()
    logger = setup_logger(
        "Inference", os.path.join(Config.WORKING_DIR, "inference.log")
    )
    device = get_device()

    # Default model path if not provided
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info(f"Starting submission generation using model: {model_path}")

    # 1. Load Data and Mappings
    # We use get_dataloaders to ensure consistent transforms and reuse cached mappings
    # We ignore train/val loaders (returned as first two elements)
    logger.info("Loading test data and taxonomy mappings...")
    _, _, test_loader, maps = get_dataloaders(load_cached_data=True)

    # 2. Initialize Model
    # pretrained=False because we are loading our own fine-tuned weights
    logger.info(f"Initializing model architecture: {Config.MODEL_NAME}")
    model = HierarchicalConvNeXt(pretrained=False)

    # 3. Load Weights
    if not os.path.exists(model_path):
        logger.error(
            f"Model file not found at {model_path}. Cannot generate submission."
        )
        return

    try:
        logger.info("Loading model weights...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    model.to(device)

    # 4. Run Inference
    logger.info("Running inference with Test-Time Augmentation (TTA)...")
    pred_indices, image_ids = predict_tta(model, test_loader, device)

    # 5. Map Indices to Original Category IDs
    # The 'idx_to_species' map converts the contiguous model index (0..N-1)
    # back to the original dataset category_id.
    # JSON keys are loaded as strings, so we convert them to ints for lookup.
    idx_to_species_raw = maps["idx_to_species"]
    idx_to_species = {int(k): int(v) for k, v in idx_to_species_raw.items()}

    final_preds = []
    for idx in pred_indices:
        if idx in idx_to_species:
            final_preds.append(idx_to_species[idx])
        else:
            # Fallback for safety, though this should not occur with correct mappings
            logger.warning(
                f"Predicted index {idx} not found in mapping. Defaulting to 0."
            )
            final_preds.append(0)

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": image_ids, "Predicted": final_preds})

    # 7. Save to CSV
    output_path = os.path.join(Config.OUTPUT_DIR, output_file)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission successfully saved to {output_path}")
