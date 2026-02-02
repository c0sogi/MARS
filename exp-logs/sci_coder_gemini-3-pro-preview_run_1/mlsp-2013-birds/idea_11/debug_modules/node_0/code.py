import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, sanitize_pseudo_labels
from library.model import BirdResNet
from library.data import get_dataloaders, Mixup
from library.training import run_swa_training, predict, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing demonstration...")

    # 1. Setup and Configuration
    # Initialize Config with debug=True to use a subset of data (50 samples) and fewer epochs (2)
    cfg = Config(debug=True)
    set_seed(cfg.SEED)

    print(f"Configuration loaded. Working directory: {cfg.WORKING_DIR}")
    print(f"Device: {cfg.DEVICE}")

    # 2. Data Loading (Teacher Stage)
    print("\n--- Step 1: Loading Data for Teacher Stage ---")
    loaders_teacher = get_dataloaders(cfg, stage="teacher")

    train_loader = loaders_teacher["train"]
    val_loader = loaders_teacher["val"]

    # Verify Data Loading
    images, targets = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Targets shape: {targets.shape}")

    # Assertions for shape correctness
    # Batch size might be smaller if dataset < batch_size, but in debug mode N=50, Batch=32
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    assert (
        images.shape[2] == cfg.IMG_HEIGHT
    ), f"Expected height {cfg.IMG_HEIGHT}, got {images.shape[2]}"
    assert (
        images.shape[3] == cfg.IMG_WIDTH
    ), f"Expected width {cfg.IMG_WIDTH}, got {images.shape[3]}"
    assert (
        targets.shape[1] == cfg.NUM_CLASSES
    ), f"Expected {cfg.NUM_CLASSES} classes, got {targets.shape[1]}"

    # Verify Mixup
    mixup_fn = Mixup(alpha=cfg.MIXUP_ALPHA)
    mixed_images, mixed_targets = mixup_fn(images, targets)
    assert mixed_images.shape == images.shape
    assert mixed_targets.shape == targets.shape
    print("Data loading and Mixup verification successful.")

    # 3. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    model = BirdResNet(
        num_classes=cfg.NUM_CLASSES, pretrained=False
    )  # False for speed in demo
    model = model.to(cfg.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_output = model(images.to(cfg.DEVICE))
    assert dummy_output.shape == (images.shape[0], cfg.NUM_CLASSES)
    print("Model initialized and forward pass verified.")

    # 4. Teacher Training
    print("\n--- Step 3: Training Teacher Model (SWA) ---")
    # Using the debug configuration parameters automatically
    teacher_save_path = cfg.TEACHER_CHECKPOINT_TEMPLATE.format(0)

    teacher_model = run_swa_training(
        cfg,
        model,
        train_loader,
        val_loader,
        epochs=cfg.TEACHER_EPOCHS,
        swa_start_epoch=cfg.TEACHER_SWA_START_EPOCH,
        save_path=teacher_save_path,
    )

    assert os.path.exists(teacher_save_path), "Teacher model checkpoint was not saved."
    print("Teacher training complete.")

    # 5. Pseudo-Label Generation
    print("\n--- Step 4: Generating Pseudo-Labels ---")
    loaders_inference = get_dataloaders(cfg, stage="inference")
    test_loader = loaders_inference["test"]

    # Predict on test set
    raw_pseudo_labels = predict(test_loader, teacher_model, cfg.DEVICE)

    # Sanitize
    clean_pseudo_labels = sanitize_pseudo_labels(raw_pseudo_labels)

    # Verify pseudo-labels shape corresponds to test set size (limited by debug MAX_SAMPLES)
    # In debug mode, test set is also capped at 50
    assert len(clean_pseudo_labels) <= cfg.MAX_SAMPLES
    assert clean_pseudo_labels.shape[1] == cfg.NUM_CLASSES
    print(f"Generated pseudo-labels for {len(clean_pseudo_labels)} samples.")

    # 6. Student Training
    print("\n--- Step 5: Training Student Model ---")
    # Initialize a fresh student model
    student_model = BirdResNet(num_classes=cfg.NUM_CLASSES, pretrained=False)
    student_model = student_model.to(cfg.DEVICE)

    # Get student dataloaders (Train + Test w/ Pseudo)
    loaders_student = get_dataloaders(
        cfg, stage="student", pseudo_labels=clean_pseudo_labels
    )
    student_train_loader = loaders_student["train"]

    # Verify dataset size increase
    # Teacher train size (approx 50) + Test size (approx 50) -> Student train size (approx 100)
    # Note: Exact numbers depend on the stratification and debug slicing, but should be > teacher size
    print(f"Student training batches: {len(student_train_loader)}")

    student_trained = run_swa_training(
        cfg,
        student_model,
        student_train_loader,
        val_loader,  # Validate on original validation set
        epochs=cfg.STUDENT_EPOCHS,
        swa_start_epoch=cfg.STUDENT_SWA_START_EPOCH,
        save_path=cfg.STUDENT_CHECKPOINT,
    )

    assert os.path.exists(
        cfg.STUDENT_CHECKPOINT
    ), "Student model checkpoint was not saved."
    print("Student training complete.")

    # 7. Submission Generation
    print("\n--- Step 6: Generating Submission ---")
    # Use the trained student model to predict on test set again for final submission
    # We reuse the inference loader
    generate_submission(cfg, student_trained, test_loader, cfg.SUBMISSION_PATH)

    assert os.path.exists(cfg.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(cfg.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check format: Id, Probability
    assert "Id" in df_sub.columns and "Probability" in df_sub.columns
    # Check that probabilities are within [0, 1]
    assert df_sub["Probability"].min() >= 0.0
    assert df_sub["Probability"].max() <= 1.0

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
