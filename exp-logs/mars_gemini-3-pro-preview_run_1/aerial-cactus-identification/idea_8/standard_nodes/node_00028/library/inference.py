import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.dataset import load_and_cache_data, CactusDataset, get_transforms
from library.model import CactusRepVGG


def predict_with_tta(model, inputs):
    """
    Performs inference using Test Time Augmentation (TTA).
    Averages predictions across 4 views: Original, H-Flip, V-Flip, Rot180.

    Args:
        model (nn.Module): The trained model (in deploy mode).
        inputs (torch.Tensor): Batch of images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probabilities (B, 1).
    """
    # Define the 4 views
    # 1. Original
    # 2. Horizontal Flip (dim 3)
    # 3. Vertical Flip (dim 2)
    # 4. Rotate 180 (H-Flip + V-Flip)
    views = [
        inputs,
        torch.flip(inputs, dims=[3]),
        torch.flip(inputs, dims=[2]),
        torch.flip(inputs, dims=[2, 3]),
    ]

    total_probs = None

    # Run inference on each view
    with torch.no_grad():
        for view in views:
            # Model output is logits
            logits = model(view)
            # Convert to probability
            probs = torch.sigmoid(logits)

            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs

    # Average the probabilities
    avg_probs = total_probs / len(views)
    return avg_probs


def generate_submission(
    model_paths,
    output_file="./submission/submission.csv",
    metadata_path="./metadata/test_metadata.csv",
    device="cuda",
    batch_size=128,
    load_cached_data=True,
    num_samples=None,
):
    """
    Generates the submission file using an ensemble of models and TTA.

    Args:
        model_paths (list): List of paths to model checkpoints (.pth files).
        output_file (str): Path to save the submission CSV.
        metadata_path (str): Path to test metadata CSV.
        device (str): Device to run inference on ('cuda' or 'cpu').
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached data.
        num_samples (int, optional): Limit number of samples for debugging.
    """
    print(f"Starting submission generation with {len(model_paths)} models...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. Load Test Data
    # load_and_cache_data handles the caching logic internally
    imgs, _, ids = load_and_cache_data(
        metadata_path, "test", load_cached_data=load_cached_data
    )

    # Optional: Subset for debugging
    if num_samples is not None:
        imgs = imgs[:num_samples]
        ids = ids[:num_samples]
        print(f"Debugging: Limiting inference to {num_samples} samples.")

    # Create Dataset and DataLoader
    # Use 'val' transforms which only apply normalization
    test_dataset = CactusDataset(imgs, transform=get_transforms("val"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # Array to store accumulated probabilities for all images
    # Shape: (N_samples,)
    ensemble_probs = np.zeros(len(test_dataset), dtype=np.float32)

    device = torch.device(device)

    # 2. Iterate through each model in the ensemble
    for i, model_path in enumerate(model_paths):
        print(
            f"Processing model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}"
        )

        # Initialize model architecture
        # Note: We initialize with deploy=False to load training weights, then switch
        model = CactusRepVGG(num_classes=1, deploy=False)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Fuse blocks for faster inference (Structural Re-parameterization)
        model.switch_to_deploy()

        model.to(device)
        model.eval()

        model_predictions = []

        # Inference loop
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                # Predict with TTA
                batch_probs = predict_with_tta(model, inputs)

                # Flatten and store
                model_predictions.append(batch_probs.cpu().numpy().ravel())

        # Concatenate predictions for this model
        full_model_preds = np.concatenate(model_predictions)

        # Add to ensemble accumulator
        ensemble_probs += full_model_preds

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    # 3. Average predictions
    avg_probs = ensemble_probs / len(model_paths)

    # 4. Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "has_cactus": avg_probs})

    # Save to CSV
    df_sub.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Total predictions: {len(df_sub)}")
    print(f"Average probability: {df_sub['has_cactus'].mean():.4f}")
