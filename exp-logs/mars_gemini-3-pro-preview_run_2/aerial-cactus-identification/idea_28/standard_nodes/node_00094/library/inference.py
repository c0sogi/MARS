import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.utils import get_device
from library.model import CustomResNet3x3
from library.dataset import CactusDataset, get_transforms


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Computes probabilities for the original images, horizontally flipped versions,
    and vertically flipped versions, then averages them.

    Args:
        model: The trained PyTorch model.
        loader: DataLoader for the test set.
        device: The computation device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle case where loader might return (images, labels) or just images
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # TTA Strategy: Original, Horizontal Flip, Vertical Flip
            # dim 3 is Width (for H-Flip), dim 2 is Height (for V-Flip)
            img_orig = images
            img_h = torch.flip(images, dims=[3])
            img_v = torch.flip(images, dims=[2])

            # Compute probabilities
            out_orig = torch.sigmoid(model(img_orig))
            out_h = torch.sigmoid(model(img_h))
            out_v = torch.sigmoid(model(img_v))

            # Average predictions
            avg_preds = (out_orig + out_h + out_v) / 3.0

            all_preds.append(avg_preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0).flatten()


def generate_submission(
    test_imgs, test_ids, model_paths, output_dir="./submission", batch_size=128
):
    """
    Generates the submission file by averaging predictions from multiple models.

    Args:
        test_imgs (np.ndarray): Test images array (N, H, W, C).
        test_ids (np.ndarray): Test image IDs.
        model_paths (list): List of paths to trained model checkpoints.
        output_dir (str): Directory to save the submission file.
        batch_size (int): Batch size for inference.
    """
    device = get_device()

    # Prepare Dataset and Loader
    # Use 'test' transforms which typically include ToTensor() (scaling 0-255 to 0-1)
    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    ensemble_preds = []

    print(f"Generating predictions using {len(model_paths)} models...")

    for path in model_paths:
        print(f"Processing model: {path}")

        # Initialize model architecture
        model = CustomResNet3x3(num_classes=1)
        model.to(device)

        # Load trained weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        # Generate predictions with TTA
        preds = predict_with_tta(model, test_loader, device)
        ensemble_preds.append(preds)

    # Average predictions across the ensemble
    final_preds = np.mean(ensemble_preds, axis=0)

    # Create submission DataFrame
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
