import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_test_loader
from library.model import get_model


def predict_fold(fold_idx, test_loader, device):
    """
    Performs inference on the test set using the model from a specific fold.
    Applies Test-Time Augmentation (TTA) by averaging predictions of the
    original and horizontally flipped images.

    Args:
        fold_idx (int): The fold index to load the checkpoint for.
        test_loader (DataLoader): The DataLoader for the test set.
        device (str): Computation device.

    Returns:
        ids (np.ndarray): Array of image IDs.
        preds (np.ndarray): Array of predicted probabilities (0-1).
    """
    # Initialize model architecture (no pretraining needed as we load weights)
    model = get_model(pretrained=False, device=device)

    # Construct checkpoint path
    checkpoint_filename = f"fold_{fold_idx}.pth"
    checkpoint_path = os.path.join(Config.checkpoint_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint for fold {fold_idx} not found at {checkpoint_path}. Skipping."
        )
        return None, None

    # Load weights
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle case where checkpoint is a dict containing state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # 1. Forward pass: Original images
            # Output is logits, apply sigmoid for probability
            logits_orig = model(images).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass: Flipped images (TTA)
            # Flip along width (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped).squeeze(1)
            probs_flip = torch.sigmoid(logits_flip)

            # 3. Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_preds.append(probs_avg.cpu().numpy())
            all_ids.append(ids.numpy())

    return np.concatenate(all_ids), np.concatenate(all_preds)


def run_inference():
    """
    Main inference routine.
    1. Loads test data.
    2. Iterates through all folds to generate predictions.
    3. Aggregates predictions via averaging.
    4. Saves the submission file.
    """
    device = Config.device
    test_loader = get_test_loader(
        batch_size=Config.batch_size, num_workers=Config.num_workers
    )

    # Dictionary to map ID to a list of predictions from different folds
    # id (int) -> [prob_fold_0, prob_fold_1, ...]
    id_to_preds = {}

    folds_processed = 0

    for fold_idx in range(Config.n_folds):
        print(f"--- Processing Fold {fold_idx} ---")
        ids, preds = predict_fold(fold_idx, test_loader, device)

        if ids is None:
            continue

        folds_processed += 1

        for img_id, pred in zip(ids, preds):
            img_id = int(img_id)
            if img_id not in id_to_preds:
                id_to_preds[img_id] = []
            id_to_preds[img_id].append(pred)

    if folds_processed == 0:
        print("Error: No valid checkpoints found. Cannot generate submission.")
        return

    print(f"Aggregating predictions from {folds_processed} folds...")

    # Prepare final data
    final_rows = []
    # Sort IDs to ensure consistent order (though CSV does not strictly require it, it's good practice)
    sorted_ids = sorted(id_to_preds.keys())

    for img_id in sorted_ids:
        preds_list = id_to_preds[img_id]
        # Arithmetic mean of probabilities
        avg_prob = sum(preds_list) / len(preds_list)
        final_rows.append({"id": img_id, "label": avg_prob})

    submission_df = pd.DataFrame(final_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.submission_path, index=False)

    print(f"Submission saved to {Config.submission_path}")
    print("First 5 rows of submission:")
    print(submission_df.head())
