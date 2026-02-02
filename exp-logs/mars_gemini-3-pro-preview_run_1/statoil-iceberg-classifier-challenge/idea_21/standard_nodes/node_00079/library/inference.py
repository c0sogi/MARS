import os
import torch
import pandas as pd
import numpy as np
from library.network import IcebergResNet
from library.config import SUBMISSION_PATH


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for a dataset using Test-Time Augmentation (TTA).

    TTA Strategy (Klein Four-Group):
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 180 (Horizontal + Vertical Flip)

    Args:
        model (nn.Module): The trained model (in eval mode).
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (ids, probabilities)
            ids (list): List of image IDs.
            probabilities (np.ndarray): Flattened array of predicted probabilities.
    """
    model.eval()
    ids_all = []
    probs_all = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for images, angles, _, ids in dataloader:
            images = images.to(device)
            angles = angles.to(device)

            # --- TTA 1: Original ---
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # --- TTA 2: Horizontal Flip ---
            # images shape: (B, 3, H, W). Width is dim 3.
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # --- TTA 3: Vertical Flip ---
            # Height is dim 2.
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # --- TTA 4: Rotate 180 (HFlip + VFlip) ---
            images_hv = torch.flip(images, [2, 3])
            out4 = model(images_hv, angles)
            prob4 = torch.sigmoid(out4)

            # Average predictions across the 4 views
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            probs_all.extend(avg_prob.cpu().numpy().flatten())
            ids_all.extend(ids)

    return ids_all, np.array(probs_all)


def ensemble_predict(model_paths, dataloader, device, output_path=SUBMISSION_PATH):
    """
    Generates predictions by averaging outputs from multiple model checkpoints.
    Handles loading of both standard and SWA models.

    Args:
        model_paths (list): List of file paths to model checkpoints (.pth).
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save the submission CSV.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'is_iceberg' predictions.
    """
    print(f"Starting ensemble prediction with {len(model_paths)} models...")

    ensemble_probs = None
    ids_list = None

    for i, path in enumerate(model_paths):
        print(f"Processing model {i+1}/{len(model_paths)}: {path}")

        # Initialize model architecture
        model = IcebergResNet()
        model = model.to(device)

        # Load checkpoint
        if not os.path.isfile(path):
            print(f"Warning: Checkpoint not found at {path}. Skipping.")
            continue

        checkpoint = torch.load(path, map_location=device)

        # Extract state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Handle SWA or DataParallel keys (strip 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load weights
        model.load_state_dict(new_state_dict)

        # Generate predictions with TTA
        ids, probs = predict_with_tta(model, dataloader, device)

        # Accumulate probabilities
        if ensemble_probs is None:
            ensemble_probs = probs
            ids_list = ids
        else:
            ensemble_probs += probs

            # Sanity check for ID alignment
            if ids_list != ids:
                raise ValueError(
                    f"ID mismatch detected in model {path}. Ensure DataLoader is deterministic."
                )

    # Average probabilities
    if ensemble_probs is not None:
        avg_probs = ensemble_probs / len(model_paths)

        # Create DataFrame
        df = pd.DataFrame({"id": ids_list, "is_iceberg": avg_probs})

        # Save to CSV
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Ensemble predictions saved to {output_path}")

        return df
    else:
        print("Error: No predictions generated (check model paths).")
        return pd.DataFrame(columns=["id", "is_iceberg"])
