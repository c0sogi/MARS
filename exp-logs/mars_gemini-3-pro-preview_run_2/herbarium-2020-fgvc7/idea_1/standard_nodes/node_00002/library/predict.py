import os
import torch
import pandas as pd
import numpy as np
from torch.cuda import amp
from library.config import Config
from library.model import get_model
from library.dataset import get_dataloaders


def generate_submission(test_loader=None):
    """
    Generates predictions for the test set using the best model and saves to CSV.

    Args:
        test_loader (DataLoader, optional): The data loader for the test set.
                                            If None, it will be created using get_dataloaders.
    """
    device = Config.DEVICE

    # Ensure we have a test_loader
    if test_loader is None:
        print("Initializing dataloaders to retrieve test set...")
        # We pass load_cached_data=True to utilize existing cache if available,
        # though strictly speaking we only need the test set here.
        _, _, test_loader = get_dataloaders(load_cached_data=True)

    print("Loading model architecture...")
    # We don't need to download ImageNet weights since we are loading a checkpoint
    model = get_model(pretrained=False)

    # Load best model weights
    checkpoint_path = Config.MODEL_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        best_f1 = checkpoint.get("best_f1", "N/A")
        print(f"Loaded model with Best F1: {best_f1}")
    else:
        print(
            f"Warning: No checkpoint found at {checkpoint_path}. Using random initialization."
        )

    model.eval()

    all_ids = []
    all_preds = []

    print(f"Starting inference on {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            # Use Automatic Mixed Precision for inference
            with amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)

            # Get predicted class indices (argmax of logits)
            preds = torch.argmax(outputs, dim=1)

            all_ids.extend(image_ids.numpy())
            all_preds.extend(preds.cpu().numpy())

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    # Sort by Id to ensure consistent ordering matching sample submission
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
