import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

from library.config import CFG
from library.model import AppleDiseaseModel


def predict_tta(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Performs inference with Test Time Augmentation (Horizontal Flip).

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        np.ndarray: Array of predicted probabilities with shape (N_samples, N_classes).
    """
    model.eval()
    probs_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Forward pass: Original images
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass: Horizontally flipped images
            # images shape is (B, C, H, W), flip on last dimension (W)
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0
            probs_list.append(avg_probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def soft_voting_ensemble(
    probs_list: List[np.ndarray], threshold: float = 0.5
) -> List[str]:
    """
    Combines predictions from multiple models using soft voting (averaging probabilities).

    Args:
        probs_list (List[np.ndarray]): List of probability arrays from different models.
        threshold (float): Threshold to convert probabilities to binary labels.

    Returns:
        List[str]: List of space-delimited label strings for submission.
    """
    # Compute mean probability across all models
    # Stack along a new axis and take mean
    avg_probs = np.mean(probs_list, axis=0)

    # Convert to binary predictions
    binary_preds = (avg_probs > threshold).astype(int)

    final_labels = []
    class_names = CFG.class_labels

    for row in binary_preds:
        # Get indices where prediction is 1
        indices = np.where(row == 1)[0]

        # Map indices to class names
        labels = [class_names[i] for i in indices]

        # Join with space
        label_str = " ".join(labels)

        # Fallback: if no class is predicted, assume "healthy"
        if not label_str:
            label_str = "healthy"

        final_labels.append(label_str)

    return final_labels


def generate_submission(
    model_configs: List[Tuple[str, str]],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """
    Runs the inference pipeline: loads models, predicts, ensembles, and saves submission.

    Args:
        model_configs (List[Tuple[str, str]]): List of (model_name, weight_path) tuples.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.
    """
    print(f"Starting inference with {len(model_configs)} models...")

    all_model_probs = []

    for model_name, weight_path in model_configs:
        print(f"Processing model: {model_name}")
        print(f"Loading weights from: {weight_path}")

        # Initialize model
        # We use pretrained=False because we are loading custom weights immediately
        model = AppleDiseaseModel(model_name=model_name, pretrained=False)

        # Load weights
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weight file not found: {weight_path}")

        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)

        # Predict with TTA
        probs = predict_tta(model, test_loader, device)
        all_model_probs.append(probs)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Ensemble
    print("Ensembling predictions...")
    final_labels = soft_voting_ensemble(all_model_probs, threshold=0.5)

    # Create Submission DataFrame
    # Retrieve image IDs from the dataset dataframe
    test_df = test_loader.dataset.df.copy()

    # Ensure alignment
    if len(test_df) != len(final_labels):
        raise ValueError(
            f"Mismatch between test set size ({len(test_df)}) and predictions ({len(final_labels)})"
        )

    test_df["labels"] = final_labels

    # Select required columns
    submission_df = test_df[["image", "labels"]]

    # Save
    os.makedirs(os.path.dirname(CFG.submission_path), exist_ok=True)
    submission_df.to_csv(CFG.submission_path, index=False)

    print(f"Submission saved successfully to {CFG.submission_path}")
    print(submission_df.head())
