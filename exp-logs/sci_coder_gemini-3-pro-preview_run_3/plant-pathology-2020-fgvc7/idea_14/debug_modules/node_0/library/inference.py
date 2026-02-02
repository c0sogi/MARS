import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_data, get_transforms, AppleDataset
from library.models import AppleNet


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Domain-Aware Test-Time Augmentation.
    TTA Strategy: Average of Original and Horizontal Flip.
    Vertical Flip and Transpose are explicitly excluded.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        tuple: (predictions numpy array, list of image_ids)
    """
    model.eval()
    all_preds = []
    all_ids = []

    print(f"Starting inference with TTA (Original + Horizontal Flip)...")

    with torch.no_grad():
        for batch_data in loader:
            # Handle different return signatures from Dataset depending on mode
            # AppleDataset in 'test' mode returns (image, image_id)
            images, ids = batch_data

            images = images.to(device)

            # 1. Forward Pass - Original
            outputs_orig = model(images)
            # Apply Softmax to main head logits
            probs_orig = torch.softmax(outputs_orig["main"], dim=1)

            # 2. Forward Pass - Horizontal Flip
            # Flip tensor along width dimension (dim 3: B, C, H, W)
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip["main"], dim=1)

            # 3. Average Predictions
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_preds.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    final_preds = np.concatenate(all_preds, axis=0)
    return final_preds, all_ids


def run_inference():
    """
    Main function to run the inference pipeline.
    1. Loads Test Metadata.
    2. Generates Teacher Predictions (EfficientNetV2-M).
    3. Generates Student Predictions (MaxViT-Small).
    4. Ensembles predictions (Average).
    5. Saves submission file.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    print("Loading Test Metadata...")
    # We use load_data with caching, though for test it's fast enough
    test_df = load_data(Config.TEST_CSV, "test_df", load_cached_data=True)
    print(f"Test set size: {len(test_df)}")

    # =========================================================================
    # Teacher Inference (EfficientNetV2-M)
    # =========================================================================
    print("\n" + "=" * 40)
    print("INFERENCE: TEACHER MODEL (EfficientNetV2-M)")
    print("=" * 40)

    # Setup Dataset & Loader
    teacher_dataset = AppleDataset(
        test_df, transforms=get_transforms("test", Config.TEACHER_IMG_SIZE), mode="test"
    )
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    teacher_model = AppleNet(Config.TEACHER_BACKBONE, pretrained=False)
    if os.path.exists(Config.TEACHER_CHECKPOINT):
        print(f"Loading weights from {Config.TEACHER_CHECKPOINT}")
        state_dict = torch.load(Config.TEACHER_CHECKPOINT, map_location=Config.DEVICE)
        teacher_model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint {Config.TEACHER_CHECKPOINT} not found. Using random weights."
        )

    teacher_model.to(Config.DEVICE)

    # Predict
    teacher_preds, image_ids = predict_with_tta(
        teacher_model, teacher_loader, Config.DEVICE
    )

    # Cleanup to save memory
    del teacher_model, teacher_loader, teacher_dataset
    torch.cuda.empty_cache()

    # =========================================================================
    # Student Inference (MaxViT-Small)
    # =========================================================================
    print("\n" + "=" * 40)
    print("INFERENCE: STUDENT MODEL (MaxViT-Small)")
    print("=" * 40)

    # Setup Dataset & Loader
    student_dataset = AppleDataset(
        test_df, transforms=get_transforms("test", Config.STUDENT_IMG_SIZE), mode="test"
    )
    student_loader = DataLoader(
        student_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    student_model = AppleNet(Config.STUDENT_BACKBONE, pretrained=False)
    if os.path.exists(Config.STUDENT_CHECKPOINT):
        print(f"Loading weights from {Config.STUDENT_CHECKPOINT}")
        state_dict = torch.load(Config.STUDENT_CHECKPOINT, map_location=Config.DEVICE)
        student_model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint {Config.STUDENT_CHECKPOINT} not found. Using random weights."
        )

    student_model.to(Config.DEVICE)

    # Predict
    student_preds, _ = predict_with_tta(student_model, student_loader, Config.DEVICE)

    # Cleanup
    del student_model, student_loader, student_dataset
    torch.cuda.empty_cache()

    # =========================================================================
    # Ensemble & Submission
    # =========================================================================
    print("\n" + "=" * 40)
    print("GENERATING ENSEMBLE SUBMISSION")
    print("=" * 40)

    # Unweighted Average Ensemble
    final_preds = (teacher_preds + student_preds) / 2.0

    # Create DataFrame
    submission_df = pd.DataFrame(final_preds, columns=Config.CLASSES)

    # Insert image_id at the beginning
    submission_df.insert(0, "image_id", image_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.FINAL_SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.FINAL_SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())
