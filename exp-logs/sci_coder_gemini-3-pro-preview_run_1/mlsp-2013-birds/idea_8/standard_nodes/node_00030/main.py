import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.dataset import get_data, BirdDataset, get_transforms
from library.pipeline import (
    train_teacher_ensemble,
    generate_sanitized_pseudo_labels,
    train_student_swa,
    generate_submission,
)
from library.engine import validate, predict


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 3. Load Data
    # Using load_cached_data=True to utilize preprocessed data in ./working if available
    print("Loading datasets...")
    train_data = get_data(train_df, load_cached_data=True, cache_prefix="train")
    val_data = get_data(val_df, load_cached_data=True, cache_prefix="val")
    test_data = get_data(test_df, load_cached_data=True, cache_prefix="test")

    # 4. Pipeline Execution

    # Stage 1: Train Teachers
    # Trains an ensemble of models on the labeled training set
    print("\n=== Stage 1: Teacher Ensemble ===")
    teachers = train_teacher_ensemble(train_data, val_data)

    # Stage 2: Pseudo-Labels
    # Generates soft labels for the test set using the teacher ensemble
    print("\n=== Stage 2: Pseudo-Label Generation ===")
    pseudo_labels = generate_sanitized_pseudo_labels(
        teachers, test_data, load_cached_data=True
    )

    # Stage 3: Student Training (SWA)
    # Trains a student model on Train + Pseudo-Labeled Test with SWA
    print("\n=== Stage 3: Student SWA Training ===")
    student_model = train_student_swa(train_data, test_data, pseudo_labels)

    # 5. Final Validation
    print("\n=== Final Validation ===")
    val_images, val_labels, val_ids = val_data

    # Create validation loader
    val_dataset = BirdDataset(
        val_images, val_labels, val_ids, transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validate
    val_loss, val_auc = validate(student_model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Get raw predictions for analysis
    preds, ids = predict(student_model, val_loader, Config.DEVICE)

    # Calculate Mean Absolute Error per sample
    # preds is (N, 19), val_labels is (N, 19)
    errors = np.abs(val_labels - preds).mean(axis=1)  # (N,)

    # Feature 1: Label Cardinality (Number of active species)
    cardinality = val_labels.sum(axis=1)

    # Feature 2: Image Brightness/Intensity
    # val_images is (N, H, W, 3) uint8. Normalize to 0-1 for correlation calculation.
    intensities = val_images.mean(axis=(1, 2, 3)) / 255.0

    # Calculate correlations
    # np.corrcoef returns matrix [[1, r], [r, 1]]
    corr_card = np.corrcoef(errors, cardinality)[0, 1]
    corr_int = np.corrcoef(errors, intensities)[0, 1]

    print(f"Correlation (Error vs Label Cardinality): {corr_card:.10f}")
    print(f"Correlation (Error vs Image Intensity): {corr_int:.10f}")

    # 7. Submission
    threshold = 0.9433543480067271
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(student_model, test_data)
    else:
        print(
            f"\nValidation metric ({val_auc}) does NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
