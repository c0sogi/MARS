import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm

# Import from library
from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataloaders
from library.model import HierarchicalMetricNet

# Initialize logger
logger = get_logger("inference")


def load_taxonomy_mapping():
    """
    Loads the taxonomy mapping from the cache to map species indices back to category IDs.

    Returns:
        dict: A dictionary mapping species_idx (int) -> category_id (int).
    """
    cache_path = os.path.join(Config.CACHE_DIR, "taxonomy_mapping.parquet")

    if not os.path.exists(cache_path):
        logger.error(
            f"Taxonomy mapping file not found at {cache_path}. Run training/dataset processing first."
        )
        raise FileNotFoundError(f"Taxonomy mapping file missing: {cache_path}")

    try:
        tax_map = pd.read_parquet(cache_path)
        # Create map: species_idx -> category_id
        idx_to_cat = dict(zip(tax_map["species_idx"], tax_map["category_id"]))
        logger.info(f"Loaded taxonomy mapping for {len(idx_to_cat)} species.")
        return idx_to_cat
    except Exception as e:
        logger.error(f"Failed to load taxonomy mapping: {e}")
        raise e


def predict_and_submit(checkpoint_path=None, output_path=None, debug=False):
    """
    Runs inference on the test set and generates a submission file.

    Args:
        checkpoint_path (str, optional): Path to the trained model weights.
                                         Defaults to Config.BEST_MODEL_PATH.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_FILE.
        debug (bool): If True, runs on a subset of data as defined in Config.
    """
    # Set defaults if not provided
    if checkpoint_path is None:
        checkpoint_path = Config.BEST_MODEL_PATH

    if output_path is None:
        output_path = Config.SUBMISSION_FILE

    device = torch.device(Config.DEVICE)

    logger.info("Step 1: preparing data loaders...")
    # We need get_dataloaders to get meta_counts for model initialization
    # and the test_loader for inference.
    _, _, test_loader, meta_counts = get_dataloaders(debug=debug, load_cached_data=True)

    logger.info("Step 2: Loading taxonomy mapping...")
    idx_to_cat = load_taxonomy_mapping()

    logger.info(f"Step 3: Initializing model structure ({Config.MODEL_NAME})...")
    model = HierarchicalMetricNet(
        num_species=meta_counts["num_species"],
        num_genera=meta_counts["num_genera"],
        num_families=meta_counts["num_families"],
    )
    model.to(device)

    logger.info(f"Step 4: Loading weights from {checkpoint_path}...")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully.")
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Using random initialization (Predictions will be random)."
        )

    model.eval()

    all_ids = []
    all_preds_cat = []

    logger.info("Step 5: Starting inference...")

    with torch.no_grad():
        # Iterate over test loader
        # Using tqdm for progress tracking in logs if running interactively,
        # though standard logging is preferred for scripts.
        for batch_idx, (images, image_ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            with autocast():
                # Forward pass
                # species_label=None triggers the ArcFaceLayer to return scaled cosine similarities
                outputs = model(images, species_label=None)

                # Get species logits (Cosine Similarity * Scale)
                logits = outputs["species"]

                # Get predicted indices (0 to num_species-1)
                preds_idx = torch.argmax(logits, dim=1).cpu().numpy()

            # Map indices back to original category_ids
            batch_preds_cat = [idx_to_cat.get(idx, 0) for idx in preds_idx]

            # Store results
            # image_ids is a tuple of strings/ints from the dataset
            all_ids.extend(list(image_ids))
            all_preds_cat.extend(batch_preds_cat)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches.")

    logger.info(f"Inference complete. Processed {len(all_ids)} images.")

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds_cat})

    # Ensure Id column format matches requirements (int)
    try:
        submission_df["Id"] = submission_df["Id"].astype(int)
    except ValueError:
        logger.warning("Could not convert Id column to integer. Keeping as is.")

    # Sort by Id to ensure consistent order
    submission_df.sort_values("Id", inplace=True)

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info("Head of submission:")
    # Using print for the head as requested to show output clearly
    print(submission_df.head())
