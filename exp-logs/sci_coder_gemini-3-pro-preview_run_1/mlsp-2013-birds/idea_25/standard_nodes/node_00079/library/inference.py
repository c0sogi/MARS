import os
import torch
import numpy as np
import pandas as pd
from library.utils import set_seed


def predict_with_tta(model, loader, device="cuda"):
    """
    Performs inference using a single model with Test-Time Augmentation (Horizontal Flip).

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset (e.g., test set).
        device (str): Device to run inference on.

    Returns:
        tuple: (rec_ids, probabilities)
            rec_ids (np.array): Array of recording IDs.
            probabilities (np.array): Array of shape (N, 19) containing predicted probabilities.
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)

            # 1. Forward pass original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass flipped (TTA)
            # Flip along width dimension (dim 3 for NCHW format: Batch, Channel, Height, Width)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

            # 3. Average probabilities
            probs_avg = (probs_orig + probs_flipped) / 2.0

            all_probs.append(probs_avg.cpu().numpy())
            # rec_ids is a tensor from the dataloader
            all_rec_ids.append(rec_ids.numpy())

    all_probs = np.vstack(all_probs)
    all_rec_ids = np.concatenate(all_rec_ids)

    return all_rec_ids, all_probs


def predict_ensemble(models, loader, device="cuda"):
    """
    Performs inference using an ensemble of models, each with TTA.
    The predictions from all models are averaged.

    Args:
        models (list): List of trained torch.nn.Module models.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run inference on.

    Returns:
        tuple: (rec_ids, probabilities)
    """
    avg_probs = None
    rec_ids = None

    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]

    for i, model in enumerate(models):
        r_ids, probs = predict_with_tta(model, loader, device=device)

        if avg_probs is None:
            avg_probs = probs
            rec_ids = r_ids
        else:
            # Verify alignment of recording IDs
            if not np.array_equal(rec_ids, r_ids):
                raise ValueError(
                    "Recording IDs mismatch between ensemble predictions. Ensure DataLoader is deterministic (shuffle=False)."
                )
            avg_probs += probs

    # Compute average
    if len(models) > 0:
        avg_probs /= len(models)

    return rec_ids, avg_probs


def generate_submission(
    rec_ids, probabilities, output_dir="./submission", filename="submission.csv"
):
    """
    Formats the predictions and saves the submission CSV file.

    The ID format is: rec_id * 100 + species_id.

    Args:
        rec_ids (np.array): Array of recording IDs.
        probabilities (np.array): Array of shape (N, 19) containing probabilities.
        output_dir (str): Directory to save the submission file.
        filename (str): Name of the submission file.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)

    submission_rows = []
    num_species = probabilities.shape[1]

    # Iterate through each recording and each species to create the flattened format
    for i, rid in enumerate(rec_ids):
        probs = probabilities[i]
        for species_id in range(num_species):
            # Construct the unique Id
            row_id = int(rid * 100 + species_id)
            prob = float(probs[species_id])

            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)

    # Sort by Id to ensure consistent ordering
    df_sub = df_sub.sort_values(by="Id").reset_index(drop=True)

    output_path = os.path.join(output_dir, filename)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return df_sub


def run_inference(
    models, loader, device="cuda", output_dir="./submission", filename="submission.csv"
):
    """
    Orchestrates the full inference pipeline: prediction (with TTA and Ensembling) and submission generation.

    Args:
        models (list or nn.Module): Model or list of models.
        loader (DataLoader): Test data loader.
        device (str): Device.
        output_dir (str): Output directory.
        filename (str): Output filename.
    """
    print("Starting inference...")
    rec_ids, probs = predict_ensemble(models, loader, device=device)

    print("Generating submission file...")
    generate_submission(rec_ids, probs, output_dir=output_dir, filename=filename)
    print("Inference complete.")
