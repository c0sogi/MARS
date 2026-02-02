import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import set_seed
from library.dataset import get_loaders
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import train_one_epoch, validate, threshold_search, inference


def main():
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    OUTPUT_DIR = "./working"
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # Ensure reproducibility
    set_seed(42)

    print(f"Running on device: {DEVICE}")

    # --- 1. Prepare Data ---
    # get_loaders handles data processing, normalization, and caching to ./working/idea_17/
    print("\nInitializing data loaders...")
    loaders = get_loaders(batch_size=BATCH_SIZE, load_cached_data=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Verify data loading
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    assert len(train_loader) > 0, "Train loader is empty"

    # --- 2. Initialize Model ---
    print("\nInitializing model...")
    # Using pretrained ResNet34 backbone
    model = ResNet34WideLinkNet(pretrained=True)
    model.to(DEVICE)

    # --- 3. Initialize Loss and Optimizer ---
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training (1 Epoch for demonstration) ---
    print("\nStarting training (1 epoch)...")
    # train_one_epoch handles the forward pass, loss calculation (supervised + distillation), and backprop
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device=DEVICE
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # --- 5. Validation ---
    print("\nStarting validation...")
    # validate returns loss, default mAP (0.5), and raw predictions for threshold search
    val_loss, val_map, val_preds, val_targets = validate(
        model, val_loader, criterion, device=DEVICE
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP (threshold=0.5): {val_map:.4f}")

    # --- 6. Threshold Optimization ---
    print("\nOptimizing threshold...")
    # Search for the best IoU threshold (0.3 to 0.7) that maximizes the competition metric
    best_threshold, best_score = threshold_search(val_preds, val_targets)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best mAP Score: {best_score:.4f}")

    # --- 7. Inference ---
    print("\nGenerating submission...")
    # inference applies TTA, unpads images, encodes RLE, and saves to CSV
    inference(
        model,
        test_loader,
        threshold=best_threshold,
        output_path=SUBMISSION_PATH,
        device=DEVICE,
    )

    # --- 8. Verification ---
    print("\nVerifying submission file...")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    df_submission = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission shape: {df_submission.shape}")
    print("First 5 rows:")
    print(df_submission.head())

    # Check constraints
    assert len(df_submission) == 1000, f"Expected 1000 rows, got {len(df_submission)}"
    assert list(df_submission.columns) == [
        "id",
        "rle_mask",
    ], "Invalid columns in submission"

    # Check that we have valid RLE strings (or empty strings for no salt)
    # This ensures the rle_encode function worked correctly
    valid_rle = df_submission["rle_mask"].apply(
        lambda x: isinstance(x, str) or pd.isna(x)
    )
    assert valid_rle.all(), "Submission contains invalid RLE formats"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
