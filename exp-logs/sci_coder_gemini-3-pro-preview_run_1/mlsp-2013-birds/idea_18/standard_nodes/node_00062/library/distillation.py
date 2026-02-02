import os
import torch
import numpy as np
import pandas as pd
from library.utils import set_seed


def generate_pseudo_labels(
    teacher_models,
    test_loader,
    device,
    output_path="./working/idea_18/pseudo_labels.parquet",
    load_cached_data=True,
):
    """
    Generates pseudo-labels for the test set using an ensemble of teacher models
    and Test-Time Augmentation (TTA).

    Args:
        teacher_models (list): List of trained PyTorch models (nn.Module).
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.
        output_path (str): Path to save the generated pseudo-labels.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: DataFrame containing rec_id and soft pseudo-labels.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Caching Mechanism
    if load_cached_data and os.path.exists(output_path):
        # print(f"Loading cached pseudo-labels from {output_path}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            # If load fails, proceed to re-compute
            pass

    # print("Generating pseudo-labels with TTA and Ensemble...")

    # Set all models to evaluation mode
    for model in teacher_models:
        model.eval()
        model.to(device)

    all_rec_ids = []
    all_probs = []

    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            # TTA: Create horizontally flipped images
            # Images are (N, C, H, W). Width is dimension 3.
            images_flipped = torch.flip(images, dims=[3])

            batch_ensemble_probs = []

            for model in teacher_models:
                # Inference on original images
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # Inference on flipped images
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

                # TTA Averaging for this model
                avg_probs_model = (probs_orig + probs_flip) / 2.0
                batch_ensemble_probs.append(avg_probs_model)

            # Stack and average across the ensemble
            # Shape: (Num_Teachers, Batch_Size, Num_Classes)
            batch_ensemble_probs = torch.stack(batch_ensemble_probs)

            # Mean across teachers -> (Batch_Size, Num_Classes)
            final_batch_probs = torch.mean(batch_ensemble_probs, dim=0)

            all_probs.append(final_batch_probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)

    # Sanity Check: Assert no NaNs
    if np.isnan(all_probs).any():
        raise ValueError("NaN values detected in generated pseudo-labels.")

    # Create DataFrame
    num_classes = all_probs.shape[1]
    cols = ["rec_id"] + [f"species_{i}" for i in range(num_classes)]

    # Prepare data dict
    data = {"rec_id": all_rec_ids}
    for i in range(num_classes):
        data[f"species_{i}"] = all_probs[:, i]

    df_pseudo = pd.DataFrame(data)

    # Save to cache
    df_pseudo.to_parquet(output_path, index=False)
    # print(f"Pseudo-labels saved to {output_path}")

    return df_pseudo
