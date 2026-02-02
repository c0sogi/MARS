import os
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import LightweightPyramidNet


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).

    TTA Strategy:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): The test data loader.
        device (torch.device): The computation device.

    Returns:
        np.ndarray: Array of predicted probabilities (N,).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is width)
            images_hflip = torch.flip(images, [3])
            logits_hflip = model(images_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # 3. Vertical Flip (dim 2 is height)
            images_vflip = torch.flip(images, [2])
            logits_vflip = model(images_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average probabilities across views
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0
            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def generate_submission(
    model_paths: list,
    output_file: str = "./submission/submission.csv",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Generates a submission file by ensembling predictions from multiple models using TTA.

    Args:
        model_paths (list): List of file paths to trained model checkpoints (.pth).
        output_file (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        seed (int): Random seed for reproducibility.
    """
    # 1. Setup
    set_seed(seed)
    device = get_device()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Generating submission using {len(model_paths)} models on {device}...")

    # 2. Load Data
    # We only need the test loader and ids
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, load_cached_data=True, seed=seed
    )

    num_samples = len(test_ids)
    accumulated_preds = np.zeros(num_samples, dtype=np.float64)

    # 3. Inference Loop
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model path {path} does not exist. Skipping.")
            continue

        print(f"Processing model: {path}")

        # Instantiate and load model
        model = LightweightPyramidNet().to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        # Predict with TTA
        preds = predict_with_tta(model, test_loader, device)

        # Accumulate
        accumulated_preds += preds

        # Clean up to save memory
        del model
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Average Predictions
    if len(model_paths) > 0:
        final_preds = accumulated_preds / len(model_paths)
    else:
        # Fallback if no models loaded (should not happen in valid run)
        final_preds = np.full(num_samples, 0.5)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # 6. Save
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(submission_df.head())
