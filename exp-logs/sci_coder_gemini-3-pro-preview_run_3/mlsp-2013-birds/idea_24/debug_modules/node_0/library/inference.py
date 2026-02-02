import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.models import get_bird_model
from library.dataset import get_datasets
from library.utils import seed_everything


def predict_with_tta(model, loader, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Applies Horizontal Shift TTA: Original + Left Shift + Right Shift.
    Averages the logits from the three views before applying sigmoid.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        tuple: (probabilities, rec_ids)
            - probabilities: np.ndarray of shape (N, num_classes)
            - rec_ids: np.ndarray of shape (N,)
    """
    model.eval()
    all_probs = []
    all_rec_ids = []

    # Shift amount: ~10% of image width (224 * 0.1 ~= 22)
    shift_pixels = 22

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)
            B, C, H, W = images.shape

            # 1. Original View
            logits_orig = model(images)

            # 2. Left Shift View
            # Shift content left, pad right with zeros (silence)
            # Corresponds to moving forward in time, losing the end
            img_left = torch.zeros_like(images)
            img_left[..., :-shift_pixels] = images[..., shift_pixels:]
            logits_left = model(img_left)

            # 3. Right Shift View
            # Shift content right, pad left with zeros (silence)
            # Corresponds to moving backward in time, losing the start
            img_right = torch.zeros_like(images)
            img_right[..., shift_pixels:] = images[..., :-shift_pixels]
            logits_right = model(img_right)

            # Average Logits (Ensemble of views)
            avg_logits = (logits_orig + logits_left + logits_right) / 3.0

            # Convert to probabilities
            probs = torch.sigmoid(avg_logits)

            all_probs.append(probs.cpu().numpy())
            all_rec_ids.append(rec_ids.numpy())

    return np.concatenate(all_probs), np.concatenate(all_rec_ids)


def load_and_average_checkpoints(
    model_name, checkpoint_paths, loader, device, num_classes=19
):
    """
    Loads multiple model checkpoints (snapshots), runs TTA inference for each,
    and averages their predicted probabilities to form a robust ensemble.

    Args:
        model_name (str): Name of the architecture (e.g., 'resnet18').
        checkpoint_paths (list): List of file paths to the .pth checkpoints.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        num_classes (int): Number of classes.

    Returns:
        tuple: (averaged_probabilities, rec_ids)
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for inference.")

    ensemble_probs = None
    final_rec_ids = None

    # Initialize model structure once
    model = get_bird_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)

    print(f"Ensembling {len(checkpoint_paths)} checkpoints for {model_name}...")

    for i, path in enumerate(checkpoint_paths):
        if not os.path.exists(path):
            print(f"Warning: Checkpoint not found at {path}. Skipping.")
            continue

        print(
            f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {os.path.basename(path)}"
        )

        # Load weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        # Run Inference with TTA
        probs, rec_ids = predict_with_tta(model, loader, device)

        # Accumulate probabilities
        if ensemble_probs is None:
            ensemble_probs = probs
            final_rec_ids = rec_ids
        else:
            # Ensure alignment of recording IDs across checkpoints
            if not np.array_equal(final_rec_ids, rec_ids):
                raise RuntimeError(
                    "Mismatch in recording IDs between checkpoints. Ensure deterministic DataLoader."
                )
            ensemble_probs += probs

    if ensemble_probs is None:
        raise RuntimeError("No valid checkpoints were processed.")

    # Average probabilities
    avg_probs = ensemble_probs / len(checkpoint_paths)

    return avg_probs, final_rec_ids


def save_submission(rec_ids, probabilities, output_path):
    """
    Formats predictions into the competition submission format and saves to CSV.
    Format:
        Id,Probability
        (rec_id * 100 + species_id), prob

    Args:
        rec_ids (np.ndarray): Array of recording IDs.
        probabilities (np.ndarray): Array of probabilities (N, num_classes).
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_rows = []
    num_classes = probabilities.shape[1]

    for i, rec_id in enumerate(rec_ids):
        row_probs = probabilities[i]
        for species_id in range(num_classes):
            # Construct unique Id
            unique_id = int(rec_id * 100 + species_id)
            prob = row_probs[species_id]

            submission_rows.append({"Id": unique_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)

    # Sort by Id to be neat (though not strictly required)
    df_sub = df_sub.sort_values("Id")

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_sub)} rows.")


def run_inference_pipeline(
    model_name,
    checkpoint_paths,
    output_path="./submission/submission.csv",
    device="cuda",
    batch_size=32,
    num_workers=2,
):
    """
    High-level function to execute the full inference pipeline:
    1. Load Test Data
    2. Ensemble Prediction (TTA + Checkpoint Averaging)
    3. Generate Submission File

    Args:
        model_name (str): Architecture name.
        checkpoint_paths (list): List of checkpoint files.
        output_path (str): Destination for submission CSV.
        device (str): Device.
        batch_size (int): Batch size for inference.
        num_workers (int): DataLoader workers.
    """
    seed_everything(42)

    # 1. Get Data
    # We use get_datasets to ensure consistent preprocessing
    # We only need the test dataset
    _, _, test_dataset = get_datasets(load_cached_data=True)

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Predict
    avg_probs, rec_ids = load_and_average_checkpoints(
        model_name, checkpoint_paths, loader, device
    )

    # 3. Save
    save_submission(rec_ids, avg_probs, output_path)
