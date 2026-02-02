import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple


def reconstruct_probabilities(binary_probs: np.ndarray) -> np.ndarray:
    """
    Reconstructs the 4-class probabilities from the binary [Rust, Scab] probabilities
    using the multi-label decomposition rules.

    Args:
        binary_probs (np.ndarray): Array of shape (N, 2) containing [P(Rust), P(Scab)].

    Returns:
        np.ndarray: Array of shape (N, 4) corresponding to [Healthy, Multiple, Rust, Scab].
    """
    # Extract binary probabilities
    p_r = binary_probs[:, 0]  # Probability of Rust
    p_s = binary_probs[:, 1]  # Probability of Scab

    # Apply decomposition formulas
    # Healthy: No Rust AND No Scab
    healthy = (1 - p_r) * (1 - p_s)

    # Multiple: Rust AND Scab
    multiple = p_r * p_s

    # Rust: Rust AND No Scab
    rust = p_r * (1 - p_s)

    # Scab: No Rust AND Scab
    scab = (1 - p_r) * p_s

    # Stack into (N, 4) array
    # Order: Healthy, Multiple, Rust, Scab (to align with submission columns)
    return np.stack([healthy, multiple, rust, scab], axis=1)


def predict_ensemble(
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    """
    Performs inference on the test set using an ensemble of models and Test Time Augmentation (TTA).

    TTA Strategy: Average predictions of the original image and a horizontal flip.
    Ensemble Strategy: Average predictions across all provided models.

    Args:
        models (List[nn.Module]): List of trained models.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        Tuple[List[str], np.ndarray]: List of image IDs and an array of aggregated binary probabilities.
    """
    # Ensure all models are in eval mode and on the correct device
    for model in models:
        model.to(device)
        model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)

            # TTA: Create horizontally flipped images
            # inputs shape: (B, C, H, W). Flip on dimension 3 (Width).
            inputs_flip = torch.flip(inputs, dims=[3])

            batch_ensemble_preds = []

            for model in models:
                # Forward pass - Original
                outputs_orig = model(inputs)
                probs_orig = torch.sigmoid(outputs_orig)

                # Forward pass - Flipped
                outputs_flip = model(inputs_flip)
                probs_flip = torch.sigmoid(outputs_flip)

                # TTA Average for this model
                probs_tta = (probs_orig + probs_flip) / 2.0

                batch_ensemble_preds.append(probs_tta)

            # Stack predictions from all models: Shape (Num_Models, Batch_Size, 2)
            batch_ensemble_preds = torch.stack(batch_ensemble_preds)

            # Ensemble Average: Shape (Batch_Size, 2)
            avg_batch_preds = torch.mean(batch_ensemble_preds, dim=0)

            all_preds.append(avg_batch_preds.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, np.concatenate(all_preds, axis=0)


def generate_submission(
    image_ids: List[str], binary_probs: np.ndarray, output_path: str
) -> None:
    """
    Generates the final submission CSV file.

    Args:
        image_ids (List[str]): List of image identifiers.
        binary_probs (np.ndarray): Binary probabilities [Rust, Scab] of shape (N, 2).
        output_path (str): Path to save the CSV file.
    """
    # Reconstruct the 4-class probabilities
    # Columns: [Healthy, Multiple, Rust, Scab]
    final_probs = reconstruct_probabilities(binary_probs)

    # Create DataFrame matching the sample_submission.csv format
    submission_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": final_probs[:, 0],
            "multiple_diseases": final_probs[:, 1],
            "rust": final_probs[:, 2],
            "scab": final_probs[:, 3],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
