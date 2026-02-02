import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, AverageMeter
from library.transforms import get_transforms
from library.dataset import CassavaDataset
from library.model import CassavaConvNeXt
from library.engine import (
    train_one_epoch,
    validate,
    generate_submission,
    EarlyStopping,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("1. Setting up configuration for demo run...")

    # Override CFG for a fast demonstration
    CFG.seed = 123
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.num_workers = 2
    CFG.train_subset_size = 20  # Use only 20 samples for training
    CFG.val_subset_size = 10  # Use only 10 samples for validation
    CFG.output_dir = "./working/demo_script_output"

    # Ensure output directory exists (CFG.setup() is called on import, but we changed the path)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set seeds
    seed_everything(CFG.seed)

    print(f"   Output directory: {CFG.output_dir}")
    print(f"   Device: {CFG.device}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n2. Verifying utilities...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)  # Sum=20, Count=2
    meter.update(20, n=1)  # Sum=40, Count=3

    assert (
        meter.avg == 40 / 3
    ), f"AverageMeter logic incorrect. Expected {40/3}, got {meter.avg}"
    assert meter.count == 3, "AverageMeter count incorrect."
    print("   AverageMeter verified.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline
    # -------------------------------------------------------------------------
    print("\n3. Setting up Data Pipeline...")

    # Get transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # Instantiate Datasets
    # Note: load_cached_data=False forces a reload based on the new subset sizes in CFG
    train_dataset = CassavaDataset(
        "train", transform=train_transforms, load_cached_data=False
    )
    val_dataset = CassavaDataset(
        "val", transform=val_transforms, load_cached_data=False
    )

    # Verify Dataset Lengths
    assert (
        len(train_dataset) == CFG.train_subset_size
    ), f"Train dataset size mismatch. Expected {CFG.train_subset_size}, got {len(train_dataset)}"
    assert (
        len(val_dataset) == CFG.val_subset_size
    ), f"Val dataset size mismatch. Expected {CFG.val_subset_size}, got {len(val_dataset)}"

    # Verify Item Structure
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Dataset should return an image tensor"
    assert isinstance(label, torch.Tensor), "Dataset should return a label tensor"
    assert img.shape == (
        3,
        CFG.image_size,
        CFG.image_size,
    ), f"Image shape mismatch. Expected (3, {CFG.image_size}, {CFG.image_size}), got {img.shape}"

    print("   Datasets instantiated and verified.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    print("   DataLoaders created.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n4. Initializing Model...")

    # Initialize model
    # We use pretrained=False to avoid downloading weights during this demo run
    model = CassavaConvNeXt(pretrained=False)
    model.to(CFG.device)

    # Dummy Forward Pass
    dummy_input = torch.randn(CFG.batch_size, 3, CFG.image_size, CFG.image_size).to(
        CFG.device
    )
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), f"Model output shape mismatch. Expected ({CFG.batch_size}, {CFG.num_classes}), got {output.shape}"

    print("   Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Running Training Loop (1 Epoch)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Train One Epoch
    train_loss, train_acc = train_one_epoch(
        epoch=0,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=CFG.device,
    )

    print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0 <= train_acc <= 1, "Training accuracy out of bounds"

    # Validate
    val_loss, val_acc = validate(
        model=model, val_loader=val_loader, criterion=criterion, device=CFG.device
    )

    print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Demonstrate Early Stopping Logic
    stopper = EarlyStopping(
        patience=2, mode="max", save_path=os.path.join(CFG.output_dir, "best_model.pth")
    )
    # Simulate improvement
    stopper(0.5, model, optimizer, epoch=0)
    assert os.path.exists(
        stopper.save_path
    ), "EarlyStopping failed to save checkpoint on improvement."

    # Simulate no improvement
    stopper(0.4, model, optimizer, epoch=1)
    assert stopper.counter == 1, "EarlyStopping counter did not increment."

    print("   Training loop and EarlyStopping verified.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n6. Generating Submission...")

    # Setup Test Dataset/Loader
    test_transforms = get_transforms("test")
    test_dataset = CassavaDataset(
        "test", transform=test_transforms, load_cached_data=False
    )

    # Limit test set for speed if it's large (though sample_submission is usually small)
    # The dataset class doesn't support subsetting test via CFG natively like train/val,
    # but we can just pass the full loader since sample_submission is ~2.6k rows.
    # For this specific demo, we rely on the speed of the model.

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    submission_dir = os.path.join(CFG.output_dir, "submission")

    # Generate Submission
    generate_submission(
        model=model,
        test_loader=test_loader,
        device=CFG.device,
        output_dir=submission_dir,
    )

    submission_file = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not created."

    # Verify Content
    df_sub = pd.read_csv(submission_file)
    assert (
        "image_id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) == len(test_dataset), "Submission row count mismatch."

    print(f"   Submission generated at {submission_file}")
    print("   Rows:", len(df_sub))
    print("   Columns:", df_sub.columns.tolist())

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
