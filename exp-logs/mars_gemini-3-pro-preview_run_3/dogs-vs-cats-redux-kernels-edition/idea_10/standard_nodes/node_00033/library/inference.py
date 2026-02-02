import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import Config
from library.utils import get_logger
from library.data_loader import get_dataloaders
from library.model_factory import create_model

# Initialize Logger
logger = get_logger("inference")


def predict_model(model, loader, device, use_tta=False):
    """
    Generates predictions for a specific model on a dataset.
    Implements Test Time Augmentation (Horizontal Flip) if enabled.

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.
        use_tta (bool): Whether to use Horizontal Flip TTA.

    Returns:
        np.array: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            if use_tta:
                # 2. Forward pass on horizontally flipped images
                # tensor.flip(dims) - dim 3 is width (B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs_avg = (probs_orig + probs_flipped) / 2.0
                all_preds.append(probs_avg.cpu().numpy())
            else:
                all_preds.append(probs_orig.cpu().numpy())

    # Concatenate all batches
    # Shape: (N, 1) -> flatten to (N,)
    return np.concatenate(all_preds).flatten()


def run_inference():
    """
    Orchestrates the full inference pipeline:
    1. Iterates through all models defined in Config.MODEL_SPECS.
    2. Loads the appropriate weights (SWA or Best).
    3. Generates predictions (with TTA).
    4. Computes the ensemble average.
    5. Saves the submission file.
    """
    device = torch.device(Config.DEVICE)
    model_predictions = {}

    # We need the test metadata to map predictions to IDs later
    # We can load it once via pandas directly to get the IDs
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Iterate over each model in the ensemble
    for model_name in Config.MODEL_SPECS.keys():
        logger.info(f"Starting inference for model: {model_name}")

        # 1. Initialize Model Architecture
        # We don't need to load pretrained ImageNet weights since we will load our own checkpoint
        model = create_model(model_name, pretrained=False)
        model.to(device)

        # 2. Determine Weight Path
        # Prioritize SWA weights if configured and available
        swa_path = os.path.join(Config.WORKING_DIR, f"{model_name}_swa.pth")
        best_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        checkpoint_path = None
        if Config.USE_SWA and os.path.exists(swa_path):
            checkpoint_path = swa_path
            logger.info(f"Loading SWA weights from {swa_path}")
        elif os.path.exists(best_path):
            checkpoint_path = best_path
            logger.info(f"Loading Best weights from {best_path}")
        else:
            logger.warning(
                f"No checkpoint found for {model_name}. Skipping this model."
            )
            continue

        # 3. Load Weights
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)

            # Fix for SWA 'module.' prefix if present
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load weights for {model_name}: {e}")
            continue

        # 4. Get DataLoader (Specific to model resolution)
        # We only need the test loader here
        _, _, test_loader = get_dataloaders(model_name, load_cached_data=True)

        # 5. Predict
        logger.info(f"Generating predictions (TTA={Config.TTA_FLIP})...")
        preds = predict_model(model, test_loader, device, use_tta=Config.TTA_FLIP)
        model_predictions[model_name] = preds

        logger.info(f"Finished inference for {model_name}. Samples: {len(preds)}")

    # --- Ensemble Aggregation ---
    if not model_predictions:
        logger.error("No predictions generated. Cannot create submission.")
        return

    logger.info("Aggregating ensemble predictions...")

    # Stack predictions: Shape (Num_Models, Num_Samples)
    stacked_preds = np.vstack(list(model_predictions.values()))

    # Calculate Arithmetic Mean
    final_preds = np.mean(stacked_preds, axis=0)

    # --- Submission Generation ---
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Create DataFrame
    # test_df is sorted by ID in metadata generation, and DataLoader is shuffle=False
    # so the order of final_preds matches test_df['id']
    submission_df = pd.DataFrame({"id": test_df["id"], "label": final_preds})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info("Submission saved successfully.")

    # Log some stats about the predictions
    logger.info(
        f"Prediction Stats - Mean: {final_preds.mean():.10f}, Std: {final_preds.std():.10f}"
    )
