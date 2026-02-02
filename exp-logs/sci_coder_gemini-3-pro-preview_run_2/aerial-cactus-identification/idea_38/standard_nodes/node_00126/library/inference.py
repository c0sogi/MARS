import os
import torch
import pandas as pd
import numpy as np
from library.utils import get_device, set_seed
from library.model import CactusResNet
from library.dataset import get_dataloaders


def load_and_prepare_model(checkpoint_path, device):
    """
    Loads a model checkpoint and moves it to the specified device.

    Args:
        checkpoint_path (str): Path to the .pth model file.
        device (torch.device): The device to load the model onto.

    Returns:
        model (nn.Module): The model in eval mode.
    """
    # Initialize the model structure.
    model = CactusResNet(num_classes=1)

    # Load the state dictionary
    # map_location ensures we can load GPU-trained models on CPU if necessary
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.to(device)
    model.eval()
    return model


def predict_ensemble(
    model_paths, test_loader=None, device=None, output_path="submission/submission.csv"
):
    """
    Loads multiple trained model checkpoints, applies Test Time Augmentation (TTA),
    aggregates predictions, and saves the submission file.

    Args:
        model_paths (list): List of file paths to the trained model checkpoints.
        test_loader (DataLoader, optional): DataLoader for the test set.
                                            If None, it will be created using library defaults.
        device (torch.device, optional): Device for inference. If None, auto-detected.
        output_path (str): Destination path for the submission CSV.
    """
    # Ensure reproducibility
    set_seed(42)

    if device is None:
        device = get_device()

    # If loader is not provided, initialize it
    if test_loader is None:
        # We only need the test_loader
        _, _, test_loader = get_dataloaders(batch_size=64, load_cached_data=True)

    print(f"Preparing to load {len(model_paths)} models...")
    models = []

    for path in model_paths:
        try:
            model = load_and_prepare_model(path, device)
            models.append(model)
            print(f"Loaded and re-parameterized: {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

    if not models:
        raise RuntimeError("No models were successfully loaded. Aborting inference.")

    print(
        "Starting ensemble inference with TTA (Original, Horizontal Flip, Vertical Flip)..."
    )

    results = {}

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Test Time Augmentation (TTA)
            # 1. Original Image
            # 2. Horizontal Flip (dim 3)
            # 3. Vertical Flip (dim 2)
            tta_variants = [
                images,
                torch.flip(images, dims=[3]),
                torch.flip(images, dims=[2]),
            ]

            # Accumulator for summed probabilities
            # Initialize with zeros
            batch_preds_sum = np.zeros(images.size(0), dtype=np.float64)

            for variant in tta_variants:
                for model in models:
                    # Forward pass
                    logits = model(variant)
                    # Apply Sigmoid to get probabilities
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    batch_preds_sum += probs

            # Calculate Arithmetic Mean
            # Total contributions = (Number of TTA variants) * (Number of Models)
            total_votes = len(tta_variants) * len(models)
            avg_preds = batch_preds_sum / total_votes

            # Store results mapped by ID
            for img_id, pred in zip(ids, avg_preds):
                results[img_id] = pred

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(list(results.items()), columns=["id", "has_cactus"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")

    return df
