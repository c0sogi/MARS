import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import BirdDataset
from library.model import get_resnet34
from library.utils import load_checkpoint


def generate_calibrated_pseudo_labels(
    teacher_checkpoint_paths, device=Config.DEVICE, load_cached_data=True
):
    """
    Generates calibrated pseudo-labels for the test set using an ensemble of teachers.
    Applies Test-Time Augmentation (Horizontal Flip) and Temperature Scaling.

    Args:
        teacher_checkpoint_paths (list): List of paths to teacher model checkpoints.
        device (torch.device): Device to run inference on.
        load_cached_data (bool): If True, attempts to load existing pseudo-labels from disk.

    Returns:
        pd.DataFrame: The generated pseudo-labels dataframe.
    """
    # Caching Mechanism
    if load_cached_data and os.path.exists(Config.PSEUDO_LABEL_PATH):
        print(f"Loading cached pseudo-labels from {Config.PSEUDO_LABEL_PATH}")
        try:
            return pd.read_parquet(Config.PSEUDO_LABEL_PATH)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print("Generating calibrated pseudo-labels...")

    # Setup Data (Mode='test' -> Resize + Normalize, no random augs)
    test_dataset = BirdDataset(metadata_path=Config.TEST_METADATA_PATH, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    num_samples = len(test_dataset)
    num_classes = Config.NUM_CLASSES

    # Accumulator for ensemble probabilities
    ensemble_probs = np.zeros((num_samples, num_classes), dtype=np.float32)
    rec_ids_list = []
    captured_rec_ids = False

    for teacher_path in teacher_checkpoint_paths:
        print(f"Inference with teacher: {teacher_path}")

        # Load Teacher Model
        model = get_resnet34(num_classes=num_classes, pretrained=False)
        load_checkpoint(teacher_path, model, device=device)
        model.to(device)
        model.eval()

        teacher_preds = []
        current_rec_ids = []

        with torch.no_grad():
            for images, _, rec_ids in test_loader:
                images = images.to(device)

                # --- Test-Time Augmentation (TTA) ---
                # 1. Original Forward Pass
                logits_orig = model(images)

                # 2. Horizontal Flip Forward Pass
                # Tensor shape: (B, C, H, W). Flip on Width (dim 3).
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)

                # --- Temperature Scaling ---
                # Apply T=1.5 before Sigmoid to soften distributions
                temp = Config.TEACHER_TEMP

                probs_orig = torch.sigmoid(logits_orig / temp)
                probs_flip = torch.sigmoid(logits_flip / temp)

                # Average TTA predictions
                batch_probs = (probs_orig + probs_flip) / 2.0

                teacher_preds.append(batch_probs.cpu().numpy())

                if not captured_rec_ids:
                    current_rec_ids.extend(rec_ids.numpy())

        # Concatenate predictions for this teacher
        teacher_preds_np = np.concatenate(teacher_preds, axis=0)

        # Accumulate into ensemble
        ensemble_probs += teacher_preds_np

        if not captured_rec_ids:
            rec_ids_list = current_rec_ids
            captured_rec_ids = True

    # Average across all teachers
    avg_probs = ensemble_probs / len(teacher_checkpoint_paths)

    # Create DataFrame
    cols = [f"species_{i}" for i in range(num_classes)]
    df_pseudo = pd.DataFrame(avg_probs, columns=cols)
    df_pseudo["rec_id"] = rec_ids_list

    # Save to Parquet (Cache)
    os.makedirs(os.path.dirname(Config.PSEUDO_LABEL_PATH), exist_ok=True)
    print(f"Saving pseudo-labels to {Config.PSEUDO_LABEL_PATH}")
    df_pseudo.to_parquet(Config.PSEUDO_LABEL_PATH, index=False)

    return df_pseudo


def predict_student(student_checkpoint_path, device=Config.DEVICE):
    """
    Generates final predictions using the student model and creates the submission file.

    Args:
        student_checkpoint_path (str): Path to the student model checkpoint.
        device (torch.device): Device to run inference on.
    """
    print(f"Generating final predictions with student: {student_checkpoint_path}")

    # Setup Data
    test_dataset = BirdDataset(metadata_path=Config.TEST_METADATA_PATH, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Student Model
    model = get_resnet34(num_classes=Config.NUM_CLASSES, pretrained=False)
    load_checkpoint(student_checkpoint_path, model, device=device)
    model.to(device)
    model.eval()

    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            # Single forward pass for Student (as per design)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, 19)
    all_rec_ids = np.array(all_rec_ids)  # (N,)

    # Format Submission
    # "rec_id" and "species" into a single "Id" column: rec_id * 100 + species_number
    submission_rows = []

    for i, rec_id in enumerate(all_rec_ids):
        probs_sample = all_probs[i]  # Shape (19,)
        for species_idx, prob in enumerate(probs_sample):
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Sort by Id
    df_submission = df_submission.sort_values("Id")

    # Cite {debug_lesson_17}: Explicitly cast Id to int to prevent float inference
    df_submission["Id"] = df_submission["Id"].astype(int)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
