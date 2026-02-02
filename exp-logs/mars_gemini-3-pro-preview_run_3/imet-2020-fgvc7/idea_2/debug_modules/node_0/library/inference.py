import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.model import ArtworkModel
from library.dataset import get_dataloaders


def predict(model, test_loader, threshold, device):
    """
    Performs inference on the test set using Test Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        threshold (float): Decision threshold for multi-label classification.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (ids, predictions)
            - ids (list): List of image IDs.
            - predictions (list): List of space-separated attribute ID strings.
    """
    model.eval()
    all_predictions = []
    all_ids = []

    print(f"Starting prediction with threshold: {threshold}")

    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(device)

            # --- Test Time Augmentation (TTA) ---

            # 1. Forward pass with original images
            with autocast():
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass with horizontally flipped images
            images_flipped = torch.flip(images, dims=[3])
            with autocast():
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # --- Post-processing ---

            # Binarize predictions based on threshold
            preds_bin = (avg_probs > threshold).cpu().numpy().astype(int)

            # Format predictions as space-separated strings
            for i, pred_row in enumerate(preds_bin):
                # Get indices where the prediction is 1 (active class)
                indices = np.where(pred_row == 1)[0]
                pred_str = " ".join(map(str, indices))

                all_predictions.append(pred_str)
                all_ids.append(batch_ids[i])

    return all_ids, all_predictions


def run_inference(
    model_path=Config.MODEL_PATH,
    threshold=Config.DEFAULT_THRESHOLD,
    output_path=Config.SUBMISSION_PATH,
    debug=Config.DEBUG,
):
    """
    Orchestrates the inference pipeline: loads model, predicts, and saves submission.

    Args:
        model_path (str): Path to the saved model weights.
        threshold (float): Optimized threshold for predictions.
        output_path (str): Path to save the submission CSV.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    device = Config.DEVICE
    print(f"Running inference on device: {device}")

    # 1. Load Data
    # We only need the test loader (3rd element)
    _, _, test_loader = get_dataloaders(debug=debug, load_cached_data=True)
    print(f"Test loader ready. Batches: {len(test_loader)}")

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = ArtworkModel(
        pretrained=False
    )  # Pretrained weights not needed, we load state_dict

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.to(device)

    # 3. Predict
    ids, predictions = predict(model, test_loader, threshold, device)

    # 4. Save Submission
    print("Generating submission DataFrame...")
    submission_df = pd.DataFrame({"id": ids, "attribute_ids": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
