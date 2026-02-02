import os
import sys
import pandas as pd
import torch
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.training import run_training_cycle
from library.inference import generate_pseudo_labels, generate_submission


def main():
    print("Initializing Demo Execution...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We override Config parameters to ensure the script runs quickly (within minutes)
    # and uses a separate working directory.

    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 32  # Use 32 samples (enough for a few batches)

    # Training Hyperparameters for Speed
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # SWA settings adapted for 2 epochs
    Config.TEACHER_SWA_START_EPOCH = 1
    Config.STUDENT_SWA_START_EPOCH = 1

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Load Metadata
    # ==========================================
    print("\nLoading Metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    print(f"Original Train size: {len(train_df)}")
    print(f"Original Val size: {len(val_df)}")
    print(f"Original Test size: {len(test_df)}")

    # ==========================================
    # 3. Teacher Training
    # ==========================================
    print("\n=== Phase 1: Teacher Training ===")

    # Create dataloaders for the teacher
    # We use the 'Texture' policy as an example
    teacher_loaders = get_dataloaders(
        train_df, val_df, test_df, teacher_policy="Texture"
    )

    # Verify loader batch size
    sample_batch, _ = next(iter(teacher_loaders["train"]))
    assert (
        sample_batch.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {sample_batch.shape[0]}"
    print("Teacher DataLoaders initialized successfully.")

    # Train the Teacher
    teacher_model_name = "teacher_demo"
    teacher_model = run_training_cycle(
        model_name=teacher_model_name,
        train_loader=teacher_loaders["train"],
        val_loader=teacher_loaders["val"],
        mixup_alpha=0.4,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        num_epochs=Config.NUM_EPOCHS,
        device=Config.DEVICE,
    )

    # Verify Checkpoint
    teacher_ckpt_path = os.path.join(
        Config.WORKING_DIR, f"{teacher_model_name}_swa.pth"
    )
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not created at {teacher_ckpt_path}"
        )
    print(f"Teacher training complete. Checkpoint saved: {teacher_ckpt_path}")

    # ==========================================
    # 4. Pseudo-Label Generation
    # ==========================================
    print("\n=== Phase 2: Pseudo-Label Generation ===")

    # Note: In DEBUG mode, get_dataloaders slices the test_df.
    # We must pass the matching sliced dataframe to generate_pseudo_labels
    # to ensure rec_ids match the predictions.
    sliced_test_df = test_df.head(Config.DEBUG_SAMPLES) if Config.DEBUG else test_df

    pseudo_labels_df = generate_pseudo_labels(
        teacher_checkpoints=[f"{teacher_model_name}_swa.pth"],
        test_loader=teacher_loaders["test"],
        test_df=sliced_test_df,
        device=Config.DEVICE,
        load_cached_data=False,  # Force generation for demo
    )

    # Validation
    assert not pseudo_labels_df.empty, "Pseudo-labels DataFrame is empty."
    assert (
        "rec_id" in pseudo_labels_df.columns
    ), "rec_id column missing in pseudo-labels."
    assert len(pseudo_labels_df) == len(
        sliced_test_df
    ), "Mismatch in pseudo-label count vs test set size."
    print("Pseudo-labels generated and verified.")

    # ==========================================
    # 5. Student Training
    # ==========================================
    print("\n=== Phase 3: Student Training ===")

    # Create dataloaders for the student
    # This combines train_df and pseudo_labels_df (which maps to test images)
    student_loaders = get_dataloaders(
        train_df,
        val_df,
        test_df,
        pseudo_labels_df=pseudo_labels_df,
        student_policy="Balanced",
    )

    # Train the Student
    student_model_name = "student_demo"
    student_model = run_training_cycle(
        model_name=student_model_name,
        train_loader=student_loaders["train"],
        val_loader=student_loaders["val"],
        mixup_alpha=0.2,
        swa_start_epoch=Config.STUDENT_SWA_START_EPOCH,
        num_epochs=Config.NUM_EPOCHS,
        device=Config.DEVICE,
    )

    # Verify Checkpoint
    student_ckpt_path = os.path.join(
        Config.WORKING_DIR, f"{student_model_name}_swa.pth"
    )
    if not os.path.exists(student_ckpt_path):
        raise FileNotFoundError(
            f"Student checkpoint not created at {student_ckpt_path}"
        )
    print(f"Student training complete. Checkpoint saved: {student_ckpt_path}")

    # ==========================================
    # 6. Final Inference & Submission
    # ==========================================
    print("\n=== Phase 4: Submission Generation ===")

    generate_submission(
        student_model=student_model,
        test_loader=student_loaders["test"],
        test_df=sliced_test_df,
        device=Config.DEVICE,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    # Basic format check
    assert (
        "Id" in sub_df.columns and "Probability" in sub_df.columns
    ), "Submission columns incorrect."
    assert (
        sub_df["Probability"].min() >= 0.0 and sub_df["Probability"].max() <= 1.0
    ), "Probabilities out of range."

    # Check ID format (rec_id * 100 + species_id)
    # We check the first ID
    first_id = sub_df.iloc[0]["Id"]
    assert first_id >= 100, "ID format seems incorrect (too small)."

    print("Submission generated and verified successfully.")
    print("\nDemo execution finished.")


if __name__ == "__main__":
    main()
