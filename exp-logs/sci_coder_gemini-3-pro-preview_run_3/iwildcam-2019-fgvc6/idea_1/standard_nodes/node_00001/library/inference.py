import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import get_model
from library.dataset import get_dataloaders


def predict_and_submit(checkpoint_path=None, output_path=None):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.MODEL_SAVE_PATH.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_PATH.
    """
    # Set defaults if not provided
    if checkpoint_path is None:
        checkpoint_path = Config.MODEL_SAVE_PATH
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    device = Config.DEVICE
    print(f"Inference device: {device}")

    # Initialize model architecture
    # We use pretrained=False here because we are about to load our own fine-tuned weights.
    # This avoids the unnecessary overhead of downloading/loading ImageNet weights.
    model = get_model(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Load model weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization."
        )

    # Switch to evaluation mode
    model.eval()

    # Get DataLoaders
    # We only need the test_loader (3rd element)
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    results = []
    print(f"Starting inference on {len(test_loader.dataset)} images...")

    # Inference loop
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Store results
            for id_val, pred_val in zip(ids, preds):
                results.append({"Id": id_val, "Predicted": pred_val})

    # Create DataFrame
    df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save submission
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(df)}")
