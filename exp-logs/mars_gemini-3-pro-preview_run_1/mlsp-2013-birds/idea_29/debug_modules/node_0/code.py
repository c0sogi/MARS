import sys
import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data import get_dataloaders, get_test_dataloader
from library.model import get_model
from library.training import run_training
from library.distillation import generate_pseudo_labels


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo Speed
    # -------------------------------------------------------------------------
    print("--- Configuring for Demo Run ---")

    # Set paths to a specific demo directory to avoid clutter/conflicts
    Config.WORKING_DIR = os.path.join("./working", "demo_run")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update checkpoint paths
    Config.TEACHER_1_CHECKPOINT = os.path.join(
        Config.WORKING_DIR, "checkpoints", "teacher_1_swa.pth"
    )
    Config.TEACHER_2_CHECKPOINT = os.path.join(
        Config.WORKING_DIR, "checkpoints", "teacher_2_swa.pth"
    )
    Config.TEACHER_3_CHECKPOINT = os.path.join(
        Config.WORKING_DIR, "checkpoints", "teacher_3_swa.pth"
    )
    Config.STUDENT_CHECKPOINT = os.path.join(
        Config.WORKING_DIR, "checkpoints", "student_swa.pth"
    )
    Config.PSEUDO_LABEL_PATH = os.path.join(
        Config.WORKING_DIR, "demo_pseudo_labels.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Optimization settings for speed
    Config.MAX_SAMPLES = 50  # Small subset of data
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_EPOCHS = 2  # Minimal epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PRETRAINED = False  # Avoid downloading weights

    # SWA settings: Start SWA at epoch 2 so it runs at least once
    Config.TEACHER_SWA_START_EPOCH = 2
    Config.STUDENT_SWA_START_EPOCH = 2

    # Set seed
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Data Loading Verification ---")

    # Load Teacher 1 Data (Policy: Linearity Bias)
    train_loader, val_loader = get_dataloaders(
        load_cached_data=False,
        teacher_policy="POLICY_TEACHER_1",
        use_pseudo_labels=False,
    )

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_HEIGHT,
            Config.IMG_WIDTH,
        ), "Incorrect Image Shape"
        assert labels.shape == (
            Config.BATCH_SIZE,
            Config.NUM_CLASSES,
        ), "Incorrect Label Shape"
        assert images.dtype == torch.float32, "Images should be float32 tensor"
        assert labels.dtype == torch.float32, "Labels should be float32 tensor"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    print("Data Loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation Verification
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Model Instantiation ---")
    model = get_model(pretrained=Config.PRETRAINED)

    # Check output shape
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model instantiated successfully.")

    # -------------------------------------------------------------------------
    # 4. Teacher Training Loop (Simulation)
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Training Teacher 1 ---")

    # We train one teacher for real to demonstrate the loop
    # Using POLICY_TEACHER_1 parameters
    mixup_alpha = Config.POLICY_TEACHER_1["mixup_alpha"]

    trained_teacher = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        save_path=Config.TEACHER_1_CHECKPOINT,
        mixup_alpha=mixup_alpha,
        patience=5,
    )

    assert os.path.exists(
        Config.TEACHER_1_CHECKPOINT
    ), "Teacher 1 checkpoint not saved!"
    print("Teacher 1 trained and saved.")

    # For the sake of the demo speed, we will duplicate Teacher 1 checkpoint
    # to Teacher 2 and Teacher 3 paths instead of retraining.
    # In a real scenario, we would re-initialize and train with different policies.
    print("Simulating Teacher 2 and 3 training by duplicating checkpoint...")
    shutil.copy(Config.TEACHER_1_CHECKPOINT, Config.TEACHER_2_CHECKPOINT)
    shutil.copy(Config.TEACHER_1_CHECKPOINT, Config.TEACHER_3_CHECKPOINT)

    assert os.path.exists(Config.TEACHER_2_CHECKPOINT)
    assert os.path.exists(Config.TEACHER_3_CHECKPOINT)

    # -------------------------------------------------------------------------
    # 5. Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Generating Pseudo-Labels ---")

    generate_pseudo_labels()

    assert os.path.exists(Config.PSEUDO_LABEL_PATH), "Pseudo-label file not created!"

    # Verify content
    df_pseudo = pd.read_parquet(Config.PSEUDO_LABEL_PATH)
    print(f"Pseudo-Label DataFrame Shape: {df_pseudo.shape}")

    # Check if number of rows matches test set subset
    # We set MAX_SAMPLES=50. The test set has 64 samples total.
    # With MAX_SAMPLES=50, we expect 50 rows (since 50 < 64).
    expected_rows = min(Config.MAX_SAMPLES, 64)
    assert (
        len(df_pseudo) == expected_rows
    ), f"Expected {expected_rows} pseudo-labels, got {len(df_pseudo)}"

    print("Pseudo-labels generated and verified.")

    # -------------------------------------------------------------------------
    # 6. Student Training (with Pseudo-Labels)
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Student Training ---")

    # Get Student Loaders (Train + Pseudo)
    train_loader_student, val_loader_student = get_dataloaders(
        load_cached_data=False,
        teacher_policy=None,  # Defaults to Balanced
        use_pseudo_labels=True,
    )

    # Verify dataset augmentation (Train size should be larger now)
    # Original train size with MAX_SAMPLES=50 is 50.
    # Pseudo test size is 50.
    # Total should be 100.
    print(f"Student Train Loader Size (Batches): {len(train_loader_student)}")

    # Initialize Student Model
    student_model = get_model(pretrained=Config.PRETRAINED)

    # Train Student
    run_training(
        model=student_model,
        train_loader=train_loader_student,
        val_loader=val_loader_student,
        swa_start_epoch=Config.STUDENT_SWA_START_EPOCH,
        save_path=Config.STUDENT_CHECKPOINT,
        mixup_alpha=Config.POLICY_BALANCED["mixup_alpha"],
        patience=5,
    )

    assert os.path.exists(Config.STUDENT_CHECKPOINT), "Student checkpoint not saved!"
    print("Student model trained and saved.")

    # -------------------------------------------------------------------------
    # 7. Final Inference / Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Generating Submission ---")

    # Load best student
    final_model = get_model(pretrained=False)
    load_checkpoint(final_model, Config.STUDENT_CHECKPOINT, device=Config.DEVICE)
    final_model.eval()

    test_loader, rec_ids = get_test_dataloader(load_cached_data=False)

    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)
            outputs = final_model(images)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Flatten for submission format: Id, Probability
    # Id = rec_id * 100 + species_id
    submission_rows = []
    for idx, rid in enumerate(rec_ids):
        probs = predictions[idx]
        for species_id, prob in enumerate(probs):
            row_id = int(rid * 100 + species_id)
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")

    # Verify submission format
    assert "Id" in df_sub.columns and "Probability" in df_sub.columns
    assert len(df_sub) == len(rec_ids) * Config.NUM_CLASSES

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    run_demo()
