import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_loaders
from library.model import get_model
from library.utils import seed_everything


def predict_all(load_cached_data=True, debug=False):
    """
    Generates predictions for the test set using the trained model.
    Implements Test Time Augmentation (TTA) if enabled in Config.

    Args:
        load_cached_data (bool): Whether to use cached hierarchy mappings for model initialization.
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # Apply debug setting to Config
    Config.DEBUG = debug

    # Set device and seed
    device = Config.DEVICE
    seed_everything(Config.SEED)

    print(f"Initializing Inference (Debug={Config.DEBUG}, TTA={Config.USE_TTA})...")

    # Retrieve DataLoaders (we only need the test_loader)
    # get_loaders handles metadata loading and transforms
    _, _, test_loader = get_loaders(load_cached_data=load_cached_data)

    # Initialize Model
    # We must use get_model to ensure the auxiliary heads (Genus/Family) are correctly sized
    # based on the hierarchy mapping, matching the trained checkpoint structure.
    model = get_model(pretrained=False, load_cached_hierarchy=load_cached_data)

    # Load the best trained checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    print(f"Starting prediction loop on {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass 1: Original Images
            # Model returns a dict of logits, we only care about 'species' for submission
            outputs = model(images)
            logits = outputs["species"]
            probs = F.softmax(logits, dim=1)

            # Test Time Augmentation (Horizontal Flip)
            if Config.USE_TTA:
                # Flip images along width dimension (dim 3 in NCHW format)
                images_flipped = torch.flip(images, dims=[3])

                outputs_flipped = model(images_flipped)
                logits_flipped = outputs_flipped["species"]
                probs_flipped = F.softmax(logits_flipped, dim=1)

                # Average the probabilities
                probs = (probs + probs_flipped) / 2.0

            # Get final predictions
            preds = torch.argmax(probs, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(image_ids)

    # Prepare Submission DataFrame
    # Convert image_ids to integers to match sample_submission.csv format
    try:
        id_col = [int(x) for x in all_ids]
    except ValueError:
        # Fallback if IDs are not numeric strings
        id_col = all_ids

    submission_df = pd.DataFrame({"Id": id_col, "Predicted": all_preds})

    # Sort by Id for consistency
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete.")
    print(f"Generated {len(submission_df)} predictions.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
