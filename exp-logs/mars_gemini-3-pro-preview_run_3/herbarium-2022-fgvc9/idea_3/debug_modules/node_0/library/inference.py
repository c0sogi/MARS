import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast

from library.config import Config
from library.model import PlantConvNeXt
from library.dataset import get_dataloader
from library.utils import set_seed


def predict_tta(model, loader, device):
    """
    Performs inference on the data loader using the provided model.
    Applies Test Time Augmentation (horizontal flip) if Config.TTA is True.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        tuple: (list of image_ids, list of predicted_labels)
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # Use Automatic Mixed Precision if enabled
            with autocast(enabled=Config.USE_AMP):
                # 1. Forward pass (Original)
                logits = model(images)
                probs = F.softmax(logits, dim=1)

                # 2. Forward pass (Horizontal Flip) - TTA
                if Config.TTA:
                    # Flip along width dimension (dim 3 for NCHW)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = F.softmax(logits_flipped, dim=1)

                    # Average probabilities
                    probs = 0.5 * (probs + probs_flipped)

            # Get predicted class index
            preds = torch.argmax(probs, dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(image_ids)

    return all_ids, all_preds


def generate_submission(
    model_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_FILE,
    debug=Config.DEBUG,
):
    """
    Loads the trained model, runs inference on the test set, and saves the submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Inference using device: {device}")

    # 2. Load Data
    print(f"Initializing Test DataLoader (Debug={debug})...")
    test_loader = get_dataloader("test", debug=debug, shuffle=False)

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = PlantConvNeXt()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Run Inference
    print(f"Starting inference (TTA={'Enabled' if Config.TTA else 'Disabled'})...")
    ids, preds = predict_tta(model, test_loader, device)

    # 5. Create Submission DataFrame
    # Ensure IDs are integers as per sample submission format
    ids = [int(x) for x in ids]

    df_submission = pd.DataFrame({"Id": ids, "Predicted": preds})

    # 6. Save Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions generated: {len(df_submission)}")
