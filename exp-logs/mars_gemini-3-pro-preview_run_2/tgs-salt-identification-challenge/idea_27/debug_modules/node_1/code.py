import os
import shutil
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import set_seed, rle_encode
from library.dataset import SaltDataset, get_transforms, rle_decode
from library.model import ResNet34WideLinkNet
from library.losses import MultiTaskLoss
from library.training import SaltTrainer
from library.inference import optimize_threshold, generate_submission

# Configuration
WORKING_DIR = "./working/demo_execution"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = WORKING_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 2
SUBSET_SIZE = 32  # Small subset for speed


def setup_directories():
    """Creates necessary directories for the demo."""
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Created working directory: {WORKING_DIR}")


def create_subset(dataset, size):
    """
    Manually slices the internal arrays of the dataset to create a small subset.
    This avoids modifying the library code while speeding up the demo.
    """
    if len(dataset) > size:
        dataset.images = dataset.images[:size]
        dataset.depths = dataset.depths[:size]
        dataset.ids = dataset.ids[:size]
        if hasattr(dataset, "masks") and dataset.masks is not None:
            dataset.masks = dataset.masks[:size]
    return dataset


def verify_rle_utilities():
    """Verifies that RLE encoding and decoding are consistent."""
    print("\n--- Verifying RLE Utilities ---")
    # Create a dummy mask: 101x101 with a square in the middle
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[40:60, 40:60] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    if not np.array_equal(mask, decoded):
        raise AssertionError("RLE Encode/Decode roundtrip failed!")
    print("RLE Encode/Decode logic verified successfully.")


def main():
    # 1. Setup
    set_seed(42)
    setup_directories()

    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    print("\n--- Initializing Datasets ---")
    # Train Dataset
    train_ds = SaltDataset(
        mode="train", transform=get_transforms("train"), cache_dir=CACHE_DIR
    )
    train_ds = create_subset(train_ds, SUBSET_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Val Dataset
    val_ds = SaltDataset(
        mode="val", transform=get_transforms("val"), cache_dir=CACHE_DIR
    )
    val_ds = create_subset(val_ds, SUBSET_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Test Dataset
    test_ds = SaltDataset(
        mode="test",
        transform=get_transforms(
            "test"
        ),  # Test uses val transforms (normalization only)
        cache_dir=CACHE_DIR,
    )
    test_ds = create_subset(test_ds, SUBSET_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Datasets initialized. Subset size: {SUBSET_SIZE}")

    # Verify Data Shapes
    sample_imgs, sample_masks, sample_depths, sample_ids = next(iter(train_loader))
    print(
        f"Sample Batch Shapes - Image: {sample_imgs.shape}, Mask: {sample_masks.shape}, Depth: {sample_depths.shape}"
    )

    if sample_imgs.shape != (BATCH_SIZE, 1, 128, 128):
        raise AssertionError(
            f"Expected image shape ({BATCH_SIZE}, 1, 128, 128), got {sample_imgs.shape}"
        )
    if sample_masks.shape != (BATCH_SIZE, 128, 128):
        raise AssertionError(
            f"Expected mask shape ({BATCH_SIZE}, 128, 128), got {sample_masks.shape}"
        )

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = ResNet34WideLinkNet(pretrained=True).to(DEVICE)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 1, 128, 128).to(DEVICE)
    with torch.no_grad():
        logits, depths = model(dummy_input)

    print(f"Model Output Shapes - Logits: {logits.shape}, Depths: {depths.shape}")
    if logits.shape != (2, 1, 128, 128):
        raise AssertionError("Model logits shape mismatch")
    if depths.shape != (2, 1):
        raise AssertionError("Model depth shape mismatch")

    # 4. Training
    print("\n--- Starting Training Demo ---")
    criterion = MultiTaskLoss(depth_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trainer = SaltTrainer(
        model=model,
        device=DEVICE,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=2,
        student_mode=False,
    )

    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise AssertionError("Training failed to produce 'best_model.pth'")
    print("Training demo completed successfully.")

    # 5. Inference & Threshold Optimization
    print("\n--- Optimizing Threshold ---")
    # Load best model
    trainer.load_checkpoint("best_model.pth")

    # Optimize threshold
    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    if not (0.0 <= best_threshold <= 1.0):
        raise AssertionError(f"Invalid threshold calculated: {best_threshold}")
    print(f"Optimal threshold found: {best_threshold}")

    # 6. Submission Generation
    print("\n--- Generating Submission ---")
    generate_submission(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        threshold=best_threshold,
        output_dir=SUBMISSION_DIR,
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file was not created.")

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    if list(df_sub.columns) != ["id", "rle_mask"]:
        raise AssertionError(f"Submission columns incorrect: {df_sub.columns}")

    if len(df_sub) != SUBSET_SIZE:
        raise AssertionError(
            f"Expected {SUBSET_SIZE} rows in submission, got {len(df_sub)}"
        )

    # 7. Utility Verification
    verify_rle_utilities()

    print("\n=== All demonstration steps completed successfully ===")


if __name__ == "__main__":
    main()
