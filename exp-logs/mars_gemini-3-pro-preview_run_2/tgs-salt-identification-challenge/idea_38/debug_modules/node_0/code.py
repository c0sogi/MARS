import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_decode, pad_image
from library.dataset import SaltDataset
from library.model import SaltNet
from library.losses import TeacherLoss, StudentLoss
from library.engine import Engine


def run_demo():
    print("=== Starting Salt Segmentation Pipeline Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring Environment...")
    Config.setup()
    set_seed(Config.SEED)

    # Override Config for rapid demonstration
    Config.BATCH_SIZE = 4
    Config.EPOCHS_TEACHER = 1
    Config.EPOCHS_STUDENT = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Define a small subset size
    SUBSET_SIZE = 12

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Subset Size: {SUBSET_SIZE}")

    # 2. Data Loading
    print("\n[2] Loading Datasets...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Instantiate Datasets (using subset_size to limit data loading)
    train_dataset = SaltDataset(
        train_df, mode="train", cache_name="demo_train", subset_size=SUBSET_SIZE
    )

    val_dataset = SaltDataset(
        val_df, mode="val", cache_name="demo_val", subset_size=SUBSET_SIZE
    )

    test_dataset = SaltDataset(
        test_df, mode="test", cache_name="demo_test", subset_size=SUBSET_SIZE
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Logic Verification: Check batch shapes
    sample_img, sample_mask, sample_depth = next(iter(train_loader))
    print(
        f"    Train Batch Shapes -> Img: {sample_img.shape}, Mask: {sample_mask.shape}, Depth: {sample_depth.shape}"
    )

    assert sample_img.shape == (
        Config.BATCH_SIZE,
        3,
        Config.PAD_SIZE,
        Config.PAD_SIZE,
    ), "Incorrect Image Shape"
    assert sample_mask.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PAD_SIZE,
        Config.PAD_SIZE,
    ), "Incorrect Mask Shape"
    assert sample_depth.shape == (Config.BATCH_SIZE, 1), "Incorrect Depth Shape"

    # 3. Model Instantiation & Logic Verification
    print("\n[3] Initializing Models...")

    # Teacher Model
    teacher_model = SaltNet(mode="teacher").to(device)
    # Student Model
    student_model = SaltNet(mode="student").to(device)

    # Verify Teacher Forward Pass
    print("    Verifying Teacher Forward Pass...")
    dummy_img = torch.randn(2, 1, Config.PAD_SIZE, Config.PAD_SIZE).to(device)
    dummy_depth = torch.randn(2, 1).to(device)

    with torch.no_grad():
        teacher_out = teacher_model(dummy_img, dummy_depth)

    assert teacher_out.shape == (
        2,
        1,
        Config.PAD_SIZE,
        Config.PAD_SIZE,
    ), "Teacher output shape mismatch"

    # Verify Student Forward Pass
    print("    Verifying Student Forward Pass...")
    with torch.no_grad():
        student_logits, student_depth_pred = student_model(dummy_img)

    assert student_logits.shape == (
        2,
        1,
        Config.PAD_SIZE,
        Config.PAD_SIZE,
    ), "Student logits shape mismatch"
    assert student_depth_pred.shape == (2, 1), "Student depth prediction shape mismatch"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Functions...")

    teacher_loss_fn = TeacherLoss()
    student_loss_fn = StudentLoss()

    # Create dummy targets
    dummy_mask_target = (
        (torch.rand(2, 1, Config.PAD_SIZE, Config.PAD_SIZE) > 0.5).float().to(device)
    )

    # Test Teacher Loss
    t_loss = teacher_loss_fn(teacher_out, dummy_mask_target)
    assert not torch.isnan(t_loss), "Teacher loss is NaN"
    print(f"    Teacher Loss Check: {t_loss.item():.4f}")

    # Test Student Loss (Labeled)
    s_loss_lbl = student_loss_fn(
        student_logits, student_depth_pred, dummy_mask_target, dummy_depth
    )
    assert not torch.isnan(s_loss_lbl), "Student Labeled loss is NaN"
    print(f"    Student Labeled Loss Check: {s_loss_lbl.item():.4f}")

    # 5. Engine Execution: Teacher Training
    print("\n[5] Training Teacher (1 Epoch)...")
    optimizer_teacher = torch.optim.Adam(
        teacher_model.parameters(), lr=Config.LEARNING_RATE
    )

    teacher_loss_epoch = Engine.train_teacher_epoch(
        teacher_model, train_loader, optimizer_teacher, device
    )
    print(f"    Teacher Epoch Loss: {teacher_loss_epoch:.4f}")

    # 6. Engine Execution: Student Training
    print("\n[6] Training Student (1 Epoch)...")
    optimizer_student = torch.optim.Adam(
        student_model.parameters(), lr=Config.LEARNING_RATE
    )

    # For demo purposes, we use the same train_loader as both labeled and unlabeled source
    # In a real scenario, unlabeled_loader would come from the test set or extra data
    student_loss_epoch = Engine.train_student_epoch(
        student_model,
        labeled_loader=train_loader,
        unlabeled_loader=train_loader,  # Mocking unlabeled with train data for demo
        optimizer=optimizer_student,
        device=device,
    )
    print(f"    Student Epoch Loss: {student_loss_epoch:.4f}")

    # 7. Validation & Threshold Optimization
    print("\n[7] Validating and Optimizing Threshold...")

    # Validate Teacher
    val_score = Engine.validate(teacher_model, val_loader, device)
    print(f"    Teacher Validation mAP: {val_score:.4f}")

    # Optimize Threshold
    best_th = Engine.optimize_threshold(teacher_model, val_loader, device)
    print(f"    Optimal Threshold found: {best_th:.2f}")

    # 8. Marginalized Pseudo-Label Generation
    print("\n[8] Generating Marginalized Pseudo-Labels...")

    # We use a list containing our single trained teacher for demonstration
    teacher_ensemble = [teacher_model]

    # Generate pseudo labels for the test subset
    pseudo_labels = Engine.generate_marginalized_pseudo_labels(
        teacher_ensemble, test_loader, device
    )

    # Verify pseudo labels
    sample_id = test_df.iloc[0]["id"]
    if sample_id in pseudo_labels:
        pl_shape = pseudo_labels[sample_id].shape
        print(f"    Pseudo-label generated for {sample_id}, shape: {pl_shape}")
        assert pl_shape == (
            1,
            Config.PAD_SIZE,
            Config.PAD_SIZE,
        ), "Pseudo-label shape mismatch"
    else:
        raise AssertionError("Pseudo-label generation failed for test sample")

    # 9. Submission Generation
    print("\n[9] Generating Submission...")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    Engine.generate_submission(
        student_model, test_loader, device, submission_path, threshold=0.5
    )

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"    Submission saved to {submission_path}")
        print(f"    Submission rows: {len(sub_df)}")
        print(f"    First few rows:\n{sub_df.head(3)}")

        # Verify format
        assert (
            "id" in sub_df.columns and "rle_mask" in sub_df.columns
        ), "Submission columns mismatch"
        assert len(sub_df) == SUBSET_SIZE, "Submission row count mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
