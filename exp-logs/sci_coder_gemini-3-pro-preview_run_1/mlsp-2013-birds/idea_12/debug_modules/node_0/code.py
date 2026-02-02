import os
import sys
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import pandas as pd
import numpy as np

# Import from the provided library
from library.configuration import Config
from library.utilities import set_seed, get_logger
from library.data_loader import get_dataloaders, get_combined_dataloader
from library.architecture import get_seresnet_model
from library.training_engine import (
    train_one_epoch,
    validate,
    update_swa,
    predict_with_tta,
    save_submission,
)


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing Configuration...")
    config = Config()

    # Modify config for a fast demonstration run
    config.update(
        BATCH_SIZE=8,  # Small batch size for speed
        TEACHER_EPOCHS=1,  # Run only 1 epoch
        STUDENT_EPOCHS=1,
        NUM_WORKERS=2,  # Minimal workers
        WORKING_DIR="./working/demo_run",
        TEACHER_CHECKPOINT_PREFIX="./working/demo_run/teacher",
        STUDENT_CHECKPOINT_PATH="./working/demo_run/student.pth",
        PSEUDO_LABELS_PATH="./working/demo_run/pseudo_labels.parquet",
        SUBMISSION_PATH="./working/demo_run/submission.csv",
    )

    # Create working directory
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Verification
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty."

    # Inspect one batch
    images, labels, indices = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Expected: (B, 3, 256, 640)
    print(f"Batch Label Shape: {labels.shape}")  # Expected: (B, 19)

    assert images.shape == (config.BATCH_SIZE, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
    assert labels.shape == (config.BATCH_SIZE, config.NUM_CLASSES)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing Model...")
    model = get_seresnet_model(config)
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            config.NUM_CLASSES,
        ), "Model output shape mismatch"

    # ==========================================
    # 4. Teacher Training (Stage 1 Demo)
    # ==========================================
    print("\nStarting Teacher Training Demo...")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize SWA Model wrapper
    swa_model = AveragedModel(model)

    # Train for one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, device, config, epoch=1)
    print(f"Epoch 1 Training Loss: {avg_loss:.4f}")
    assert avg_loss > 0, "Training loss should be positive"

    # Validate
    metrics = validate(model, val_loader, device, config)
    print(f"Validation Metrics: {metrics}")
    assert "score" in metrics and 0 <= metrics["score"] <= 1, "Invalid validation score"

    # Update SWA (Demonstration)
    update_swa(swa_model, model)
    print("SWA Model updated.")

    # ==========================================
    # 5. Pseudo-Labeling (Stage 2 Demo)
    # ==========================================
    print("\nGenerating Pseudo-Labels...")
    # Use the trained model (or swa_model) to predict on test set
    # Using TTA (Test Time Augmentation)
    test_probs = predict_with_tta(swa_model, test_loader, device)

    assert test_probs.shape == (
        len(test_loader.dataset),
        config.NUM_CLASSES,
    ), "Test prediction shape mismatch"

    # Create pseudo-label dataframe
    # We need rec_id and file_path from test metadata to construct the dataframe
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Construct the pseudo-label dataframe matching train.csv structure
    pseudo_df = df_test.copy()

    # Fill in the species columns with soft labels (probabilities)
    # Note: In a real scenario, one might threshold these or use soft labels directly.
    # The data loader expects 'species_X' columns.
    for i in range(config.NUM_CLASSES):
        pseudo_df[f"species_{i}"] = test_probs[:, i]

    # Save to parquet
    pseudo_df.to_parquet(config.PSEUDO_LABELS_PATH)
    print(f"Pseudo-labels saved to {config.PSEUDO_LABELS_PATH}")
    assert os.path.exists(config.PSEUDO_LABELS_PATH), "Pseudo-label file not created"

    # ==========================================
    # 6. Student Training (Stage 3 Demo)
    # ==========================================
    print("\nStarting Student Training with Combined Data...")

    # Get combined loader (Train + Pseudo-Test)
    combined_loader = get_combined_dataloader(config, config.PSEUDO_LABELS_PATH)
    print(f"Combined Batches: {len(combined_loader)}")
    assert len(combined_loader) >= len(
        train_loader
    ), "Combined loader should be larger or equal to train loader"

    # Initialize a fresh student model
    student_model = get_seresnet_model(config)
    student_model.to(device)
    student_optimizer = optim.AdamW(student_model.parameters(), lr=config.LEARNING_RATE)

    # Train student for one epoch
    student_loss = train_one_epoch(
        student_model, combined_loader, student_optimizer, device, config, epoch=1
    )
    print(f"Student Training Loss: {student_loss:.4f}")

    # ==========================================
    # 7. Inference and Submission
    # ==========================================
    print("\nGenerating Submission...")

    # Predict on test set using the student model
    final_probs = predict_with_tta(student_model, test_loader, device)

    # Save submission
    save_submission(final_probs, config.TEST_METADATA_PATH, config.SUBMISSION_PATH)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"
    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Check submission format
    # Expected rows: N_test_samples * 19 classes
    expected_rows = len(test_loader.dataset) * config.NUM_CLASSES
    print(f"Submission Rows: {len(df_sub)} (Expected: {expected_rows})")
    assert len(df_sub) == expected_rows, "Submission row count mismatch"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns mismatch"

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    main()
