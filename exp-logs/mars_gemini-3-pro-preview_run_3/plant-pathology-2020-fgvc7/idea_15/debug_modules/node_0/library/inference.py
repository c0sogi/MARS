import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.modeling import AppleNet
from library.utils import seed_everything


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    Strictly avoids vertical flips or transpositions to respect gravity priors.

    Args:
        model (nn.Module): The trained model (loaded with EMA weights).
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: The averaged probability predictions (N, Num_Classes).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Forward pass (Original)
            # Use Automatic Mixed Precision for consistency and efficiency
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

            # 2. Forward pass (Horizontal Flip)
            # Flip along width dimension (dim=3 for BCHW)
            images_flipped = torch.flip(images, dims=[3])

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits_flipped = model(images_flipped)
                probs_flipped = torch.softmax(logits_flipped, dim=1)

            # Average probabilities (Ensemble over augmentations)
            avg_probs = (probs + probs_flipped) / 2.0

            preds.append(avg_probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def generate_submission(model_paths, test_loader, output_path):
    """
    Loads multiple models, performs inference with TTA, averages predictions,
    and saves the submission file.

    Args:
        model_paths (list): List of paths to trained model checkpoints.
        test_loader (DataLoader): DataLoader for the test set.
        output_path (str): Path to save the submission CSV.
    """
    device = torch.device(Config.DEVICE)
    seed_everything(Config.SEED)

    print(f"Starting inference with {len(model_paths)} models...")

    ensemble_preds = None
    # Extract image IDs from the dataset dataframe
    image_ids = test_loader.dataset.df["image_id"].values

    for model_path in model_paths:
        filename = os.path.basename(model_path)

        # Determine architecture from filename based on Config.BACKBONES
        arch_name = None
        for b in Config.BACKBONES:
            if b in filename:
                arch_name = b
                break

        if arch_name is None:
            print(
                f"Warning: Could not determine architecture for {filename}. Skipping."
            )
            continue

        print(f"Processing model: {arch_name} (File: {filename})")

        # Initialize model structure
        # We use pretrained=False because we are about to load specific trained weights
        model = AppleNet(
            model_name=arch_name, num_classes=Config.NUM_CLASSES, pretrained=False
        )

        # Load weights
        # Note: The saved checkpoints from training already contain the EMA weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights for {filename}: {e}")
            continue

        model.to(device)

        # Generate predictions with TTA
        preds = predict_with_tta(model, test_loader, device)

        if ensemble_preds is None:
            ensemble_preds = preds
        else:
            ensemble_preds += preds

    if ensemble_preds is None:
        print("Error: No predictions generated. Submission file not created.")
        return

    # Average predictions across all models in the ensemble
    ensemble_preds /= len(model_paths)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(ensemble_preds, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", image_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
