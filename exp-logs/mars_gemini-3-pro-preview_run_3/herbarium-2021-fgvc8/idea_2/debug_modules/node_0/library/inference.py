import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.network import PlantClassifier
from library.data_loader import get_dataloaders


def generate_submission(
    test_loader=None, model_path=None, output_path=None, device=None
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        test_loader (DataLoader, optional): DataLoader for the test set.
                                            If None, it will be created using get_dataloaders.
        model_path (str, optional): Path to the trained model checkpoint.
                                    If None, checks for SWA model then best model in working dir.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_PATH.
        device (torch.device, optional): Device to run inference on. Defaults to Config.DEVICE.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup Configuration
    if device is None:
        device = torch.device(Config.DEVICE)

    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # 2. Prepare DataLoader
    if test_loader is None:
        # We retrieve the test loader from the library function.
        # load_cached_data=True allows reusing cached weights if they exist (though primarily for train).
        _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Determine Model Checkpoint
    if model_path is None:
        swa_path = os.path.join(Config.WORKING_DIR, "model_swa.pth")
        best_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

        # Prioritize SWA model if configured and available
        if Config.USE_SWA and os.path.exists(swa_path):
            print(f"Inference: Loading SWA model from {swa_path}")
            model_path = swa_path
        elif os.path.exists(best_path):
            print(f"Inference: Loading best standard model from {best_path}")
            model_path = best_path
        else:
            raise FileNotFoundError(
                f"No suitable model checkpoint found in {Config.WORKING_DIR}. "
                "Ensure training has completed successfully."
            )

    # 4. Initialize Model
    # We use pretrained=False because we are loading our own fine-tuned weights.
    print(f"Initializing model architecture: {Config.MODEL_NAME}")
    model = PlantClassifier(pretrained=False)
    model = model.to(device)

    # 5. Load State Dict
    print(f"Loading weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}")

    model.eval()

    # 6. Inference Loop
    ids = []
    predictions = []

    print(f"Starting inference on {len(test_loader.dataset)} samples...")

    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            # Use Automatic Mixed Precision if enabled in Config
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)

            # Get Class Predictions (Argmax of logits)
            preds = torch.argmax(outputs, dim=1)

            # Store results
            # image_ids are returned as tensors by the loader
            ids.extend(image_ids.numpy())
            predictions.extend(preds.cpu().numpy())

    # 7. Generate Submission File
    df_submission = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")

    return df_submission
