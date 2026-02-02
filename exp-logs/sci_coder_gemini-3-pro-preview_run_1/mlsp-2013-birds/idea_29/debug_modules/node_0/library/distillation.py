import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.model import get_model
from library.utils import load_checkpoint
from library.data import get_test_dataloader


def generate_pseudo_labels():
    """
    Executes Stage 2 of the pipeline: Calibrated Pseudo-Label Generation.

    1. Loads the 3 SWA Teacher models.
    2. Performs inference on the Test set (Fold 1) with TTA (Horizontal Flip).
    3. Applies Temperature Scaling to logits.
    4. Averages probabilities across the ensemble.
    5. Saves soft labels to a Parquet file for Student training.
    """
    device = Config.DEVICE
    print(f"Generating Pseudo-Labels on {device}...")

    # -------------------------------------------------------------------------
    # 1. Load Teacher Ensemble
    # -------------------------------------------------------------------------
    teacher_paths = [
        Config.TEACHER_1_CHECKPOINT,
        Config.TEACHER_2_CHECKPOINT,
        Config.TEACHER_3_CHECKPOINT,
    ]

    teachers = []
    for i, path in enumerate(teacher_paths):
        print(f"Loading Teacher {i+1} from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

        # Initialize model architecture (Homogeneous Ensemble)
        model = get_model(
            device=device, pretrained=False, num_classes=Config.NUM_CLASSES
        )

        # Load weights
        load_checkpoint(model, path, device=device)
        model.eval()
        teachers.append(model)

    # -------------------------------------------------------------------------
    # 2. Prepare Data
    # -------------------------------------------------------------------------
    # Load test data (Fold 1). We use the cache if available.
    test_loader, rec_ids = get_test_dataloader(load_cached_data=True)

    # Storage for ensemble predictions
    ensemble_probs = []

    # -------------------------------------------------------------------------
    # 3. Inference Loop (with TTA and Calibration)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Test-Time Augmentation: Horizontal Flip
            # Shape: (B, C, H, W)
            images_flipped = torch.flip(images, dims=[3])

            batch_probs_sum = torch.zeros(
                (images.size(0), Config.NUM_CLASSES), device=device, dtype=torch.float32
            )

            for model in teachers:
                # Forward pass original
                logits_orig = model(images)

                # Forward pass flipped
                logits_flip = model(images_flipped)

                # TTA Averaging (Logit space)
                logits_avg = (logits_orig + logits_flip) / 2.0

                # Temperature Scaling (Calibrate confidence)
                scaled_logits = logits_avg / Config.TEMPERATURE

                # Convert to Probability
                probs = torch.sigmoid(scaled_logits)

                batch_probs_sum += probs

            # Ensemble Averaging
            # Average the probabilities across the 3 teachers
            avg_probs = batch_probs_sum / len(teachers)

            ensemble_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    if len(ensemble_probs) > 0:
        final_probs = np.concatenate(ensemble_probs, axis=0)
    else:
        # Handle empty dataset edge case
        final_probs = np.zeros((0, Config.NUM_CLASSES))

    # -------------------------------------------------------------------------
    # 4. Sanitization and Saving
    # -------------------------------------------------------------------------
    # Assert no NaNs
    if np.isnan(final_probs).any():
        raise ValueError("NaN values detected in generated pseudo-labels!")

    # Verify shape alignment
    if len(final_probs) != len(rec_ids):
        raise ValueError(
            f"Mismatch between predictions ({len(final_probs)}) and recording IDs ({len(rec_ids)})"
        )

    print(f"Generated pseudo-labels for {len(final_probs)} samples.")

    # Construct DataFrame
    # Columns: rec_id, species_0, species_1, ...
    data = {"rec_id": rec_ids}
    for i in range(Config.NUM_CLASSES):
        data[f"species_{i}"] = final_probs[:, i]

    df_pseudo = pd.DataFrame(data)

    # Save to Parquet
    save_path = Config.PSEUDO_LABEL_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_pseudo.to_parquet(save_path, index=False)
    print(f"Pseudo-labels saved to {save_path}")

    # Optional: Print stats for verification
    mean_conf = final_probs.mean()
    max_conf = final_probs.max()
    print(
        f"Pseudo-Label Stats - Mean Confidence: {mean_conf:.4f}, Max Confidence: {max_conf:.4f}"
    )
