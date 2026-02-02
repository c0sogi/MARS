import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_loader import get_dataloaders
from library.model import AsymmetryGatedSiameseNetwork

logger = get_logger("inference")


def predict_test_set(
    model_path=None,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    save_path=Config.SUBMISSION_PATH,
):
    """
    Runs inference on the test set, aggregates predictions by prediction_id,
    and generates the submission file.

    Args:
        model_path (str): Path to the trained model weights. Defaults to best_model.pth in cache.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
        debug (bool): Whether to run in debug mode (subsampled data).
        debug_sample_size (int): Number of samples to use in debug mode.
        save_path (str): Path to save the submission CSV.
    """
    seed_everything(Config.SEED)

    # 1. Determine Model Path
    if model_path is None:
        model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Starting inference using model: {model_path}")

    # 2. Load Data
    # We only need the test loader here.
    # get_dataloaders handles metadata loading and caching internally.
    _, _, test_loader = get_dataloaders(
        train_batch_size=batch_size,  # Not used but required by signature
        val_batch_size=batch_size,
        load_cached_data=True,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # 3. Initialize Model
    model = AsymmetryGatedSiameseNetwork()

    # Load weights
    # map_location ensures we can load a GPU model on CPU if needed
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    logger.info("Running prediction loop...")
    with torch.no_grad():
        for batch_idx, (target_img, contra_img, prediction_ids) in enumerate(
            test_loader
        ):
            # Move inputs to device
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)

            # Forward pass
            logits = model(target_img, contra_img)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            # prediction_ids is a tuple of strings from the dataloader
            # Zip them with probabilities
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "prob": prob})

    # 5. Aggregation
    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        logger.warning("No predictions generated. Check data loader or debug settings.")
        # Create empty submission with correct columns just in case
        df_submission = pd.DataFrame(columns=["prediction_id", "cancer"])
    else:
        # Group by prediction_id and take the MAX probability
        # A patient might have multiple views (CC, MLO) for the same breast (prediction_id).
        # We take the maximum likelihood of cancer among available views.
        logger.info("Aggregating predictions (Max pooling over views)...")
        df_submission = df_results.groupby("prediction_id")["prob"].max().reset_index()
        df_submission.columns = ["prediction_id", "cancer"]

    # 6. Save Submission
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_submission.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
    logger.info(f"Submission shape: {df_submission.shape}")

    # Print first few rows for verification
    logger.info("First 5 rows of submission:")
    logger.info(f"\n{df_submission.head().to_string()}")

    return df_submission
