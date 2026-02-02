import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import get_model
from library.utils import load_checkpoint
from library.dataset import get_dataloaders


def run_inference(model, loader, device=Config.DEVICE, use_tta=False):
    """
    Runs inference on the provided loader using the given model.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The data loader (validation or test).
        device (str): Device to run inference on.
        use_tta (bool): If True, applies horizontal flip Test-Time Augmentation.

    Returns:
        tuple: (rec_ids, probs)
            rec_ids (np.ndarray): Array of recording IDs.
            probs (np.ndarray): Array of predicted probabilities (N, NumClasses).
    """
    model.eval()
    model.to(device)

    all_rec_ids = []
    all_probs = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)

            # Forward Pass 1: Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward Pass 2: Horizontal Flip (Time Inversion)
                # Images are (B, C, H, W). Flip on width (dim 3).
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average Probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_rec_ids.append(rec_ids.numpy())

    return np.concatenate(all_rec_ids), np.concatenate(all_probs)


def create_submission_file(rec_ids, probs, output_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the competition submission format and saves to CSV.

    Args:
        rec_ids (np.ndarray): Array of recording IDs.
        probs (np.ndarray): Array of probabilities (N, 19).
        output_path (str): Path to save the submission CSV.
    """
    submission_rows = []

    # Ensure strict integer types for IDs
    rec_ids = rec_ids.astype(int)

    num_classes = probs.shape[1]

    for i, rid in enumerate(rec_ids):
        row_probs = probs[i]
        for species_idx in range(num_classes):
            # Construct Id: rec_id * 100 + species_id
            submission_id = rid * 100 + species_idx
            probability = row_probs[species_idx]

            submission_rows.append({"Id": submission_id, "Probability": probability})

    df = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)


def generate_pseudo_labels(rec_ids, probs, threshold=None):
    """
    Formats predictions into a DataFrame suitable for merging as pseudo-labels.

    Args:
        rec_ids (np.ndarray): Recording IDs.
        probs (np.ndarray): Probabilities.
        threshold (float, optional): If provided, binarizes predictions.
                                     Otherwise keeps soft labels.

    Returns:
        pd.DataFrame: DataFrame with columns ['rec_id', 'species_0', ..., 'species_18']
    """
    cols = ["rec_id"] + [f"species_{i}" for i in range(probs.shape[1])]

    data = np.column_stack((rec_ids, probs))
    df = pd.DataFrame(data, columns=cols)

    # rec_id must be int
    df["rec_id"] = df["rec_id"].astype(int)

    if threshold is not None:
        # Apply threshold to species columns
        species_cols = [c for c in cols if c.startswith("species_")]
        df[species_cols] = (df[species_cols] >= threshold).astype(int)

    return df


def predict_and_submit(
    checkpoint_path, output_path=Config.SUBMISSION_PATH, use_tta=False
):
    """
    End-to-end inference helper: loads model, predicts on test set, saves submission.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_path (str): Path to save the submission file.
        use_tta (bool): Whether to use Test-Time Augmentation.
    """
    # Load Data (Only test loader needed)
    _, _, test_loader = get_dataloaders(
        train_metadata=Config.TRAIN_METADATA,
        val_metadata=Config.VAL_METADATA,
        test_metadata=Config.TEST_METADATA,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    # Use pretrained=False to avoid downloading weights, as we load from checkpoint
    model = get_model(
        pretrained=False, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
    )
    load_checkpoint(model, checkpoint_path, device=Config.DEVICE)

    # Inference
    rec_ids, probs = run_inference(
        model, test_loader, device=Config.DEVICE, use_tta=use_tta
    )

    # Submit
    create_submission_file(rec_ids, probs, output_path)
