import os
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from library.config import Config
from library.utils import seed_everything


def predict_tta(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[int]]:
    """
    Performs Test Time Augmentation (TTA) inference on the provided dataloader.

    Strategy:
    1. Forward pass on the original image.
    2. Forward pass on the horizontally flipped image.
    3. Average the softmax probabilities.
    4. Determine the class with the maximum average probability.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the test dataset.
        device: The compute device (CPU or GPU).

    Returns:
        ids: List of image IDs.
        preds: List of predicted category integers.
    """
    model.eval()
    ids = []
    preds = []

    # Ensure reproducibility for any non-deterministic ops (though TTA here is deterministic)
    seed_everything(Config.SEED)

    with torch.no_grad():
        for images, _, img_ids in dataloader:
            images = images.to(device, non_blocking=True)

            # 1. Original View
            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # 2. Flipped View (Horizontal Flip)
            # Input is (Batch, Channels, Height, Width). Flip on Width (dim 3).
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped)
            probs_flip = torch.softmax(logits_flip, dim=1)

            # 3. Average Probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            # 4. Argmax to get class
            batch_preds = torch.argmax(avg_probs, dim=1)

            # Store results
            ids.extend(img_ids)
            preds.extend(batch_preds.cpu().numpy())

    return ids, preds


def generate_submission(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = Config.SUBMISSION_PATH,
) -> None:
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the test dataset.
        device: The compute device.
        save_path: File path to save the submission CSV.
    """
    print(f"Starting TTA Inference on {len(dataloader.dataset)} images...")

    # Run Inference
    ids, predictions = predict_tta(model, dataloader, device)

    # Create DataFrame
    # Note: Using 'Category' as the column name to match the provided sample_submission.csv
    submission_df = pd.DataFrame({"Id": ids, "Category": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Total rows: {len(submission_df)}")
    print(submission_df.head())
