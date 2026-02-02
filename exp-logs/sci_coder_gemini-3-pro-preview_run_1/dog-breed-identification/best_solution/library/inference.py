import torch
import pandas as pd
import os
import numpy as np
from library.config import Config
from library.model import DogClassifier
from library.utils import get_checkpoint_path
from library.trainer import predict


def predict_fold(model, loader, device):
    """
    Performs inference on a single fold's model using the provided loader.
    Applies Test Time Augmentation (TTA) if configured.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Probabilities for the test set.
    """
    # Use the predict function from library.trainer which handles the loop and TTA logic
    return predict(model, loader, device, use_tta=Config.use_tta)


def generate_submission(test_loader, model_paths, classes, device):
    """
    Aggregates predictions from multiple model checkpoints (Ensemble) and generates a submission file.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        model_paths (list): List of filenames for the model checkpoints.
        classes (list): List of class names corresponding to the model outputs.
        device (torch.device): Device to run inference on.
    """
    # Initialize accumulator for probabilities
    # We don't know the exact size yet, but predict returns (N, C)
    avg_probs = None

    print(f"Starting inference on {len(model_paths)} models...")

    for i, ckpt_name in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {ckpt_name}")

        # Instantiate the model structure
        model = DogClassifier(num_classes=len(classes), pretrained=False)

        # Load the checkpoint
        full_path = get_checkpoint_path(ckpt_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Checkpoint not found at {full_path}")

        state_dict = torch.load(full_path, map_location=device)

        # Handle SWA/AveragedModel state dict keys
        # AveragedModel usually wraps the model in 'module', so keys look like 'module.backbone...'
        # We need to strip 'module.' to match DogClassifier's structure.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "n_averaged":
                continue

            if k.startswith("module."):
                new_key = k[7:]  # Remove 'module.'
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        # Generate predictions for this fold
        fold_probs = predict_fold(model, test_loader, device)

        if avg_probs is None:
            avg_probs = fold_probs
        else:
            avg_probs += fold_probs

    # Average the probabilities
    if avg_probs is not None:
        avg_probs /= len(model_paths)
    else:
        raise RuntimeError("No predictions were generated.")

    # Prepare submission DataFrame
    # test_loader.dataset is DogDataset, which has 'ids' attribute in test mode
    test_ids = test_loader.dataset.ids

    print("Generating submission DataFrame...")
    submission_df = pd.DataFrame(avg_probs.numpy(), columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save to CSV
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)

    print(f"Submission saved to {Config.submission_path}")
    print(f"Submission shape: {submission_df.shape}")
