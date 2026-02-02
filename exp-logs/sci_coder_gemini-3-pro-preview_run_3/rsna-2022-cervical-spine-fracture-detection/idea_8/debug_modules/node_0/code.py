import os
import sys
import torch
import pandas as pd
import numpy as np
import glob
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss_score
from library.dicom_utils import load_scan
from library.data import CervicalSpineDataset, get_transforms
from library.model import BoxGuidedMILModel
from library.loss import HybridHierarchicalLoss
from library.engine import train_one_epoch, validate, inference


def main():
    print("=== Starting RSNA Cervical Spine Fracture Detection Demo ===")

    # 1. Setup and Config Overrides for Speed
    print("\n[1] Setting up configuration and environment...")
    seed_everything(42)

    # Override Config for a fast demonstration run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LEN = 16  # Reduced from 64
    Config.IMAGE_SIZE = (128, 128)  # Reduced from 256
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.BACKBONE = "resnet18"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Verify DICOM Loading (library.dicom_utils)
    print("\n[2] Verifying DICOM Loading logic...")
    # Read one entry from metadata to get a valid path
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    if len(train_meta) > 0:
        sample_row = train_meta.iloc[0]
        sample_path = os.path.join(Config.INPUT_DIR, sample_row["image_path"])

        print(f"Loading scan from: {sample_path}")
        # Load scan
        volume = load_scan(sample_path, resize_to=Config.IMAGE_SIZE)

        print(f"Volume shape: {volume.shape}")
        # Check shape: (Depth, H, W)
        assert len(volume.shape) == 3, "Volume should be 3D (Depth, H, W)"
        assert volume.shape[1] == Config.IMAGE_SIZE[0], "Height mismatch"
        assert volume.shape[2] == Config.IMAGE_SIZE[1], "Width mismatch"
        assert volume.dtype == np.float32, "Volume should be float32"
        # Check value range (should be normalized 0-1)
        if volume.size > 0:
            assert (
                0.0 <= volume.min() and volume.max() <= 1.0
            ), "Volume values should be in [0, 1]"
    else:
        print("Warning: Train metadata is empty. Skipping DICOM load check.")

    # 3. Verify Dataset and DataLoaders (library.data)
    print("\n[3] Verifying Dataset and DataLoaders...")

    # Initialize Datasets
    # We limit the dataset size to a tiny subset to ensure the script runs in < 1 hour
    subset_size = 4

    train_ds = CervicalSpineDataset(
        mode="train", transform=get_transforms("train"), load_cached_data=False
    )
    # Hack: Slice the internal dataframe to limit processing
    train_ds.df = train_ds.df.head(subset_size)
    print(f"Train dataset subset size: {len(train_ds)}")

    val_ds = CervicalSpineDataset(
        mode="val", transform=get_transforms("val"), load_cached_data=False
    )
    val_ds.df = val_ds.df.head(subset_size)
    print(f"Val dataset subset size: {len(val_ds)}")

    test_ds = CervicalSpineDataset(
        mode="test", transform=get_transforms("test"), load_cached_data=False
    )
    test_ds.df = test_ds.df.head(subset_size)
    print(f"Test dataset subset size: {len(test_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Fetch one batch to verify shapes
    images, targets, box_targets, has_box, study_ids = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")  # (B, Seq, 3, H, W)
    print(f"Batch Targets Shape: {targets.shape}")  # (B, 8)
    print(f"Batch Box Targets: {box_targets.shape}")  # (B, Seq, 7)

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert targets.shape == (Config.BATCH_SIZE, 8)
    assert box_targets.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 7)

    # 4. Verify Model (library.model)
    print("\n[4] Verifying Model Architecture...")
    model = BoxGuidedMILModel(backbone_name=Config.BACKBONE, pretrained=False).to(
        device
    )

    # Run dummy forward pass
    dummy_input = images.to(device)
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")  # Should be (B, Seq, 7)
    assert output.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 7)

    # 5. Verify Loss (library.loss)
    print("\n[5] Verifying Loss Function...")
    criterion = HybridHierarchicalLoss().to(device)

    dummy_targets = targets.to(device)
    dummy_box_targets = box_targets.to(device)
    dummy_has_box = has_box.to(device)

    loss, metrics = criterion(output, dummy_targets, dummy_box_targets, dummy_has_box)

    print(f"Calculated Loss: {loss.item()}")
    print(f"Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert "loss_mil" in metrics
    assert "loss_box" in metrics

    # 6. Verify Training Engine (library.engine)
    print("\n[6] Verifying Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    avg_loss, avg_mil, avg_box = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print(
        f"Train Epoch Done. Avg Loss: {avg_loss:.4f}, MIL: {avg_mil:.4f}, Box: {avg_box:.4f}"
    )

    # 7. Verify Validation Engine
    print("\n[7] Verifying Validation Loop...")
    val_loss, val_score = validate(model, val_loader, criterion, device)
    print(f"Validation Done. Loss: {val_loss:.4f}, Score: {val_score:.4f}")

    # 8. Verify Inference Engine
    print("\n[8] Verifying Inference and Submission...")
    # The inference function writes to Config.SUBMISSION_PATH
    inference(model, test_loader, device)

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {sub_df.shape}")
        print(sub_df.head())

        # Basic check on submission format
        assert "row_id" in sub_df.columns
        assert "fractured" in sub_df.columns
        assert (
            len(sub_df) == 14536
        ), f"Expected 14536 rows in submission (matching sample_submission), got {len(sub_df)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
