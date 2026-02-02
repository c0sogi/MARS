import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader, Subset

# Import provided library modules
from library.utils import set_seed
from library.dataset import SaltDataset, get_transforms
from library.models import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import SaltEngine

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # --------------------------
    print("Initializing Salt Segmentation Demo...")
    SEED = 42
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WORKING_DIR = "./working/demo_execution"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Hyperparameters for demo
    BATCH_SIZE = 4
    SUBSET_SIZE = 16  # Small subset for speed
    LR = 1e-4

    # 2. Data Loading
    # --------------------------
    print(f"\n[1/6] Loading Datasets (Subset size: {SUBSET_SIZE})...")

    # Initialize full datasets
    # Note: The library handles caching in ./working automatically.
    train_ds_full = SaltDataset(mode="train", transform=get_transforms("train"))
    val_ds_full = SaltDataset(mode="val", transform=get_transforms("val"))

    # Create subsets to ensure the demo runs quickly
    indices = list(range(SUBSET_SIZE))
    train_subset = Subset(train_ds_full, indices)
    val_subset = Subset(val_ds_full, indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Verify Data Shapes
    # SaltDataset returns: image, mask, depth, id
    images, masks, depths, ids = next(iter(train_loader))

    # Expected: (B, 1, 128, 128) for images/masks (after padding), (B, 1) for depths
    assert images.shape == (
        BATCH_SIZE,
        1,
        128,
        128,
    ), f"Image shape incorrect: {images.shape}"
    assert masks.shape == (
        BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask shape incorrect: {masks.shape}"
    assert depths.shape == (BATCH_SIZE, 1), f"Depth shape incorrect: {depths.shape}"
    print("Data shapes verified.")

    # 3. Model & Loss Initialization
    # --------------------------
    print("\n[2/6] Initializing Model and Loss...")

    model = ResNet34WideLinkNet().to(device)
    loss_fn = CombinedLoss(bce_weight=0.5, lovasz_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = model(images.to(device), depths.to(device))
        assert dummy_out.shape == (
            BATCH_SIZE,
            1,
            128,
            128,
        ), "Model output shape mismatch"
    print("Model initialized and verified.")

    # 4. Teacher Training Loop
    # --------------------------
    print("\n[3/6] Training Teacher (1 Epoch)...")

    engine = SaltEngine(model, device, optimizer, scheduler)

    teacher_loss = engine.train_teacher_epoch(train_loader, loss_fn)
    print(f"Teacher Train Loss: {teacher_loss:.4f}")

    # Validate
    val_loss, val_map = engine.validate(val_loader, loss_fn)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")

    # 5. Pseudo-Label Generation (Test-Time Augmentation)
    # --------------------------
    print("\n[4/6] Generating Pseudo-Labels for Student Training...")

    # Load Test Data Subset
    test_ds_full = SaltDataset(mode="test", transform=get_transforms("val"))
    test_subset = Subset(test_ds_full, indices)  # Same indices 0-15
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict probabilities (returns dict {id: np_array})
    # This uses TTA (Test Time Augmentation) internally in the engine
    pseudo_labels_dict = engine.predict_proba(test_loader)

    assert len(pseudo_labels_dict) == SUBSET_SIZE, "Pseudo-label count mismatch"
    print(f"Generated {len(pseudo_labels_dict)} pseudo-labels.")

    # 6. Student Training Loop (Noisy Student)
    # --------------------------
    print("\n[5/6] Training Student (1 Epoch)...")

    # Create Student Dataset
    # We pass the pseudo_labels dictionary. The dataset will look up masks by ID.
    # We use 'student' transforms for strong augmentation (ElasticTransform, Dropout, etc.)
    student_ds = SaltDataset(
        mode="test",
        transform=get_transforms("student"),
        pseudo_labels=pseudo_labels_dict,
    )

    # Important: We must subset the student dataset to the same indices we predicted for,
    # otherwise the dataset will try to load IDs for which we have no pseudo-labels.
    student_subset = Subset(student_ds, indices)
    student_loader = DataLoader(
        student_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Run one epoch of student training
    # This combines supervised loss (on train_loader) and consistency loss (on student_loader)
    student_loss = engine.train_student_epoch(train_loader, student_loader, loss_fn)
    print(f"Student Train Loss: {student_loss:.4f}")

    # 7. Submission Generation
    # --------------------------
    print("\n[6/6] Generating Submission File...")

    submission_path = os.path.join(WORKING_DIR, "submission_demo.csv")
    engine.generate_submission(test_loader, submission_path, threshold=0.5)

    # Verify Output
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission saved to {submission_path}")
    print(f"Submission rows: {len(df_sub)}")
    print("Head of submission:")
    print(df_sub.head())

    # Final assertion on format
    assert list(df_sub.columns) == ["id", "rle_mask"], "Submission columns mismatch"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
