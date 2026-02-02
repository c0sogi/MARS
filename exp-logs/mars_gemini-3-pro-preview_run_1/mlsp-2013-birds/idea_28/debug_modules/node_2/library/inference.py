import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.model import BirdResNet34
from library.utils import load_checkpoint


def predict_logits_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    Returns averaged logits.
    """
    model.eval()
    all_logits = []
    all_rec_ids = []

    # Ensure we capture rec_ids to align predictions correctly
    # The loader is expected to return (images, labels)
    # We rely on the loader's order. To be safe, we can assume the loader
    # iterates sequentially over the dataframe provided to it.
    # The BirdDataset doesn't return rec_id, so we assume the loader is not shuffled
    # (which is true for val/test loaders in library.data).

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Forward pass (Original)
            logits_orig = model(images)

            # 2. Forward pass (Flipped)
            # Flip along width dimension (dim 3 for NCHW)
            images_flipped = torch.flip(images, [3])
            logits_flip = model(images_flipped)

            # 3. Average Logits
            avg_logits = (logits_orig + logits_flip) / 2.0

            all_logits.append(avg_logits.cpu())

    return torch.cat(all_logits, dim=0)


def generate_pseudo_labels(
    teacher_checkpoints, test_loader, test_df, device, load_cached_data=True
):
    """
    Generates pseudo-labels for the test set using an ensemble of teachers.
    Applies Temperature Scaling and averages probabilities.

    Args:
        teacher_checkpoints (list): List of filenames for teacher models in WORKING_DIR.
        test_loader (DataLoader): DataLoader for the test set.
        test_df (pd.DataFrame): Metadata for the test set (to get rec_ids).
        device (str): Device to run inference on.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame containing rec_id and species probabilities.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "pseudo_labels.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading pseudo-labels from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating pseudo-labels from scratch...")

    num_samples = len(test_df)
    num_classes = Config.NUM_CLASSES

    # Accumulator for probabilities
    sum_probs = torch.zeros((num_samples, num_classes), dtype=torch.float32)

    # 2. Ensemble Inference
    for checkpoint_name in teacher_checkpoints:
        print(f"Processing Teacher: {checkpoint_name}")

        # Load Model
        model = BirdResNet34(pretrained=False).to(device)
        load_checkpoint(model, checkpoint_name, device=device)

        # Get Logits (with TTA)
        logits = predict_logits_with_tta(model, test_loader, device)

        # Apply Temperature Scaling
        # T = 1.5 softens the distribution
        scaled_logits = logits / Config.TEMPERATURE

        # Convert to Probabilities
        probs = torch.sigmoid(scaled_logits)

        sum_probs += probs

    # 3. Average Probabilities
    avg_probs = sum_probs / len(teacher_checkpoints)

    # 4. Sanitize (Check for NaNs)
    if torch.isnan(avg_probs).any():
        print("Warning: NaNs detected in pseudo-labels. Replacing with 0.")
        avg_probs = torch.nan_to_num(avg_probs, nan=0.0)

    # 5. Construct DataFrame
    # Columns: rec_id, species_0, species_1, ...
    avg_probs_np = avg_probs.numpy()

    pseudo_data = {"rec_id": test_df["rec_id"].values}
    for i in range(num_classes):
        pseudo_data[f"species_{i}"] = avg_probs_np[:, i]

    pseudo_df = pd.DataFrame(pseudo_data)

    # 6. Save to Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    pseudo_df.to_parquet(cache_path, index=False)
    print(f"Pseudo-labels saved to {cache_path}")

    return pseudo_df


def generate_submission(
    student_model, test_loader, test_df, device, output_path=Config.SUBMISSION_PATH
):
    """
    Generates the final submission CSV using the student model.

    Args:
        student_model (nn.Module): Trained student model.
        test_loader (DataLoader): DataLoader for the test set.
        test_df (pd.DataFrame): Metadata for the test set.
        device (str): Device.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating final submission...")

    # 1. Get Logits (with TTA)
    # Student uses standard T=1.0 for final inference
    logits = predict_logits_with_tta(student_model, test_loader, device)

    # 2. Convert to Probabilities
    probs = torch.sigmoid(logits).numpy()

    # 3. Format Submission
    # Format: Id, Probability
    # Id = rec_id * 100 + species_id

    submission_rows = []
    rec_ids = test_df["rec_id"].values

    for idx, rec_id in enumerate(rec_ids):
        row_probs = probs[idx]
        for species_id, p in enumerate(row_probs):
            submission_id = int(rec_id * 100 + species_id)
            submission_rows.append({"Id": submission_id, "Probability": p})

    submission_df = pd.DataFrame(submission_rows)

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
