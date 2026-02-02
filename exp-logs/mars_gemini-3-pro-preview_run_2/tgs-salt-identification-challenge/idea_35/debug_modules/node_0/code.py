import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import prepare_data, SaltDataset, get_transforms
from library.models import SaltNet
from library.losses import LovaszHingeLoss, StudentLoss
from library.engine import (
    train_one_epoch,
    validate,
    optimize_threshold,
    generate_submission,
)
from library.utils import set_seed


def main():
    print("Starting Salt Segmentation Library Demonstration...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for speed
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.N_FOLDS = 2
    Config.DEPTH_SCAN_RANGE = [0.0]  # Single depth scan for speed

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    print("\n--- Preparing Data ---")
    # Load data (this handles caching automatically)
    data_containers = prepare_data(load_cached_data=True)

    # Slice data to a tiny subset for demonstration speed
    subset_size = 32
    print(f"Slicing datasets to {subset_size} samples for demo...")

    for mode in ["train", "val", "test"]:
        for key in data_containers[mode]:
            if data_containers[mode][key] is not None:
                data_containers[mode][key] = data_containers[mode][key][:subset_size]

    # Verify slicing
    assert len(data_containers["train"]["images"]) == subset_size
    assert len(data_containers["test"]["ids"]) == subset_size

    # Create DataLoaders for the subset
    train_ds = SaltDataset(
        data_containers["train"]["images"],
        masks=data_containers["train"]["masks"],
        depths=data_containers["train"]["depths"],
        transform=get_transforms("train"),
        mode="train",
    )
    val_ds = SaltDataset(
        data_containers["val"]["images"],
        masks=data_containers["val"]["masks"],
        depths=data_containers["val"]["depths"],
        transform=get_transforms("val"),
        mode="val",
    )
    test_ds = SaltDataset(
        data_containers["test"]["images"],
        ids=data_containers["test"]["ids"],
        depths=data_containers["test"]["depths"],  # Needed for Teacher inference
        transform=get_transforms("test"),
        mode="test",
    )

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("DataLoaders initialized.")

    # =========================================================================
    # 3. Model & Loss Verification
    # =========================================================================
    print("\n--- Verifying Model & Loss ---")

    # Instantiate Teacher Model (Depth-Injected)
    # pretrained=False to avoid download overhead for demo
    teacher_model = SaltNet(use_depth=True, aux_head=False, pretrained=False).to(device)

    # Dummy Input
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    dummy_depth = torch.randn(2, 1).to(device)

    # Forward Pass
    teacher_out = teacher_model(dummy_img, depth=dummy_depth)
    assert teacher_out.shape == (
        2,
        1,
        128,
        128,
    ), f"Teacher output shape mismatch: {teacher_out.shape}"
    print("Teacher Model forward pass successful.")

    # Instantiate Student Model (Aux Head)
    student_model = SaltNet(use_depth=False, aux_head=True, pretrained=False).to(device)
    student_out, aux_out = student_model(dummy_img)
    assert student_out.shape == (
        2,
        1,
        128,
        128,
    ), f"Student mask output mismatch: {student_out.shape}"
    assert aux_out.shape == (2, 1), f"Student aux output mismatch: {aux_out.shape}"
    print("Student Model forward pass successful.")

    # Verify Loss
    criterion_teacher = LovaszHingeLoss()
    dummy_target = (torch.rand(2, 1, 128, 128) > 0.5).float().to(device)
    loss_val = criterion_teacher(teacher_out, dummy_target)
    assert not torch.isnan(loss_val), "Lovasz Loss returned NaN"
    print("Lovasz Loss calculation successful.")

    # =========================================================================
    # 4. Simulation: Stage 1 (Teacher Training)
    # =========================================================================
    print("\n--- Simulating Stage 1: Teacher Training ---")

    optimizer_teacher = torch.optim.AdamW(teacher_model.parameters(), lr=1e-4)

    # Train 1 epoch
    loss = train_one_epoch(
        teacher_model,
        train_loader,
        optimizer_teacher,
        device,
        epoch=1,
        criterion=criterion_teacher,
        is_student=False,
    )
    assert loss > 0, "Training loss should be positive"

    # Validate
    val_map = validate(teacher_model, val_loader, device, is_student=False)
    print(f"Teacher Validation mAP: {val_map:.4f}")

    # Save Teacher
    teacher_path = os.path.join(Config.WORKING_DIR, "demo_teacher.pth")
    torch.save(teacher_model.state_dict(), teacher_path)

    # =========================================================================
    # 5. Simulation: Stage 2 (Marginalization / Pseudo-Labeling)
    # =========================================================================
    print("\n--- Simulating Stage 2: Marginalization ---")

    # In a real run, we would scan multiple depths. Here we scan the single configured depth.
    teacher_model.eval()
    accumulated_probs = []

    with torch.no_grad():
        for data in test_loader:
            imgs = data["image"].to(device)
            # Use the depth from the dataset (which is 0.0 based on our slice logic or actuals)
            # For marginalization, we usually force specific depths.
            # Let's force the depth to 0.0 as per Config override
            d = torch.zeros((imgs.size(0), 1), device=device)

            logits = teacher_model(imgs, depth=d)
            probs = torch.sigmoid(logits).cpu().numpy()
            accumulated_probs.append(probs)

    soft_masks = np.concatenate(accumulated_probs, axis=0)
    assert soft_masks.shape == (subset_size, 1, 128, 128)
    print("Soft masks generated.")

    # =========================================================================
    # 6. Simulation: Stage 3 (Student Distillation)
    # =========================================================================
    print("\n--- Simulating Stage 3: Student Distillation ---")

    # Setup Student Data (Labeled + Unlabeled)
    # For demo, we just use the train_loader as labeled and create a pseudo loader
    pseudo_ds = SaltDataset(
        data_containers["test"]["images"],
        masks=soft_masks.squeeze(1),  # Remove channel dim for dataset
        depths=np.zeros(subset_size),
        transform=get_transforms("train"),
        mode="train",
    )
    pseudo_loader = DataLoader(pseudo_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    optimizer_student = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    criterion_student = StudentLoss()

    # Train 1 epoch
    # Labeled Step
    loss_lab = train_one_epoch(
        student_model,
        train_loader,
        optimizer_student,
        device,
        epoch=1,
        criterion=criterion_student,
        is_student=True,
        is_pseudo=False,
    )

    # Unlabeled Step
    loss_unlab = train_one_epoch(
        student_model,
        pseudo_loader,
        optimizer_student,
        device,
        epoch=1,
        criterion=criterion_student,
        is_student=True,
        is_pseudo=True,
    )

    print(f"Student Losses - Labeled: {loss_lab:.4f}, Unlabeled: {loss_unlab:.4f}")

    # Validate Student
    val_map_student = validate(student_model, val_loader, device, is_student=True)
    print(f"Student Validation mAP: {val_map_student:.4f}")

    # =========================================================================
    # 7. Final Inference & Submission
    # =========================================================================
    print("\n--- Final Inference ---")

    # Optimize Threshold
    best_thresh = optimize_threshold(student_model, val_loader, device)

    # Generate Submission
    generate_submission(student_model, test_loader, device, threshold=best_thresh)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(df_sub) == subset_size
    ), f"Submission rows {len(df_sub)} != subset size {subset_size}"
    assert "id" in df_sub.columns and "rle_mask" in df_sub.columns

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
