import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the library modules can be imported
sys.path.append(".")

from library.utils import set_seed
from library.dataset import create_dataloaders, BirdDataset, get_transforms
from library.model import get_seresnet_model
from library.pipeline import (
    train_teachers,
    generate_pseudo_labels,
    train_student,
    generate_submission,
)


def main():
    print("Starting verification of library components...")

    # 1. Setup
    set_seed(42)
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Constants
    BATCH_SIZE = 4
    NUM_CLASSES = 19
    HEIGHT = 256
    WIDTH = 640
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {DEVICE}")

    # 2. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, pseudo_labels_df=None, num_workers=2, seed=42
    )

    # Fetch one batch
    images, labels, rec_ids = next(iter(train_loader))

    # Assertions
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    assert images.shape == (
        BATCH_SIZE,
        3,
        HEIGHT,
        WIDTH,
    ), f"Expected image shape {(BATCH_SIZE, 3, HEIGHT, WIDTH)}, got {images.shape}"
    assert labels.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    ), f"Expected label shape {(BATCH_SIZE, NUM_CLASSES)}, got {labels.shape}"
    assert isinstance(rec_ids, torch.Tensor), "rec_ids should be a Tensor"

    print("Data Loading verification passed.")

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = get_seresnet_model(num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
    model.eval()

    # Move batch to device
    images = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)

    print(f"Model output shape: {outputs.shape}")

    assert outputs.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    ), f"Expected output shape {(BATCH_SIZE, NUM_CLASSES)}, got {outputs.shape}"

    print("Model Architecture verification passed.")

    # 4. Verify Pipeline: Teacher Training
    # We monkey-patch the WORKING_DIR in library.pipeline to use our demo dir
    import library.pipeline

    library.pipeline.WORKING_DIR = DEMO_DIR

    print("\n--- Verifying Teacher Training (1 Teacher, 1 Epoch) ---")
    # Train 1 teacher for 1 epoch for speed
    teacher_paths = train_teachers(
        num_teachers=1, epochs=1, batch_size=BATCH_SIZE, lr=1e-3, seed=42
    )

    assert len(teacher_paths) == 1, "Should return 1 teacher path"
    assert os.path.exists(
        teacher_paths[0]
    ), f"Teacher checkpoint not found at {teacher_paths[0]}"

    print("Teacher Training verification passed.")

    # 5. Verify Pipeline: Pseudo-label Generation
    print("\n--- Verifying Pseudo-label Generation ---")
    pseudo_labels_df = generate_pseudo_labels(
        model_paths=teacher_paths,
        output_filename="demo_pseudo_labels.parquet",
        batch_size=BATCH_SIZE,
        seed=42,
        load_cached_data=False,  # Force generation
    )

    print("Pseudo-labels DataFrame head:")
    print(pseudo_labels_df.head())

    # Assertions
    # Test set size is 64
    assert (
        len(pseudo_labels_df) == 64
    ), f"Expected 64 pseudo-labels, got {len(pseudo_labels_df)}"
    assert "rec_id" in pseudo_labels_df.columns, "rec_id column missing"

    # Check probability range
    prob_cols = [c for c in pseudo_labels_df.columns if c.startswith("species_")]
    assert len(prob_cols) == NUM_CLASSES, f"Expected {NUM_CLASSES} species columns"

    probs = pseudo_labels_df[prob_cols].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities must be between 0 and 1"

    print("Pseudo-label Generation verification passed.")

    # 6. Verify Pipeline: Student Training
    print("\n--- Verifying Student Training (1 Epoch) ---")
    student_path = train_student(
        pseudo_labels_df=pseudo_labels_df,
        student_name="demo_student",
        epochs=1,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        seed=42,
    )

    assert os.path.exists(
        student_path
    ), f"Student checkpoint not found at {student_path}"
    print("Student Training verification passed.")

    # 7. Verify Pipeline: Submission Generation
    print("\n--- Verifying Submission Generation ---")
    submission_output = os.path.join(DEMO_DIR, "demo_submission.csv")
    generate_submission(
        model_path=student_path,
        output_path=submission_output,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    assert os.path.exists(submission_output), "Submission file not created"

    # Check submission content
    sub_df = pd.read_csv(submission_output)
    print("Submission DataFrame head:")
    print(sub_df.head())

    # Expected rows: 64 test samples * 19 classes = 1216 rows
    expected_rows = 64 * NUM_CLASSES
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "Id",
        "Probability",
    ], "Incorrect columns in submission"

    print("Submission Generation verification passed.")

    print("\nAll verifications passed successfully!")


if __name__ == "__main__":
    main()
