import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.utils import set_seed
from library.dataset import SaltDataset
from library.model import ResNet34WideLinkNet
from library.losses import BCEWithLovaszLoss
from library.engine import fit
from library.inference import optimize_threshold, generate_submission

# Constants
WORKING_DIR = "./working/demo_run"
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def main():
    print(f"Starting execution on device: {DEVICE}")

    # 1. Setup and Reproducibility
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Dataset and DataLoader Instantiation
    print("\n--- Initializing Datasets ---")
    # Using debug=True to load only 50 samples for speed
    train_dataset = SaltDataset(mode="train", debug=True)
    val_dataset = SaltDataset(mode="val", debug=True)

    # Check dataset length
    print(f"Train dataset size (debug): {len(train_dataset)}")
    print(f"Val dataset size (debug): {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Verify Data Loading Logic
    print("\n--- Verifying Data Loading ---")
    sample_imgs, sample_masks, sample_depths, sample_ids = next(iter(train_loader))

    # Expected shapes:
    # Images: [B, 1, 128, 128] (After transforms/padding)
    # Masks: [B, 1, 128, 128]
    # Depths: [B, 1]
    print(f"Sample Image Shape: {sample_imgs.shape}")
    print(f"Sample Mask Shape: {sample_masks.shape}")
    print(f"Sample Depth Shape: {sample_depths.shape}")

    assert (
        sample_imgs.ndim == 4 and sample_imgs.shape[1] == 1
    ), "Image tensor shape incorrect"
    assert (
        sample_masks.ndim == 4 and sample_masks.shape[1] == 1
    ), "Mask tensor shape incorrect"
    assert (
        sample_depths.ndim == 2 and sample_depths.shape[1] == 1
    ), "Depth tensor shape incorrect"

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = ResNet34WideLinkNet(pretrained=True)
    model.to(DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = model(sample_imgs.to(DEVICE), sample_depths.to(DEVICE))

    print(f"Model Output Shape: {dummy_out.shape}")
    assert (
        dummy_out.shape == sample_masks.shape
    ), "Model output shape does not match mask shape"

    # 5. Loss Function, Optimizer, Scheduler
    print("\n--- Setup Training Components ---")
    criterion = BCEWithLovaszLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Verify Loss Calculation
    dummy_loss = criterion(dummy_out, sample_masks.to(DEVICE))
    print(f"Initial Dummy Loss: {dummy_loss.item():.4f}")
    assert not torch.isnan(dummy_loss), "Loss is NaN"

    # 6. Training Loop (Engine)
    print("\n--- Starting Training Loop ---")
    best_val_loss = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=EPOCHS,
        patience=2,  # Short patience for demo
        save_dir=WORKING_DIR,
    )
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

    # 7. Threshold Optimization
    print("\n--- Optimizing Threshold ---")
    # Load best model
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    assert 0.0 < best_threshold < 1.0, f"Threshold {best_threshold} out of bounds"

    # 8. Submission Generation
    print("\n--- Generating Submission ---")
    # Initialize test dataset (debug=True for speed)
    test_dataset = SaltDataset(mode="test", debug=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    generate_submission(
        model=model,
        test_loader=test_loader,
        threshold=best_threshold,
        device=DEVICE,
        save_path=SUBMISSION_PATH,
    )

    # Verify Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"
    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Check format
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing"
    # Since we used debug=True for test set (50 samples), check count
    assert (
        len(df_sub) == 50
    ), f"Expected 50 predictions in debug mode, got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
