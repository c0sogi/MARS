import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Import provided library modules
from library import config, utils, dataset, network, engine


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup & Reproducibility
    utils.seed_everything(seed=42)
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Create Data Subsets for Speed
    # We create small CSVs in ./working to avoid processing the full dataset
    # This ensures the demo runs in seconds/minutes rather than hours.
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(config.VAL_METADATA_PATH)
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Slice top N samples (enough for a batch)
    subset_size = 8
    train_subset_df = full_train_df.head(subset_size).copy()
    val_subset_df = full_val_df.head(subset_size).copy()
    test_subset_df = full_test_df.head(subset_size).copy()

    # Save subsets to working dir
    subset_train_path = os.path.join(config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(config.WORKING_DIR, "val_subset.csv")
    subset_test_path = os.path.join(config.WORKING_DIR, "test_subset.csv")

    train_subset_df.to_csv(subset_train_path, index=False)
    val_subset_df.to_csv(subset_val_path, index=False)
    test_subset_df.to_csv(subset_test_path, index=False)

    # Monkey-patch config paths to point to our subsets
    # This allows us to use library functions that rely on config paths
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path

    print(f"Created subsets with {subset_size} samples each.")

    # 3. Preprocessing (Compute Best Slices)
    # We manually call utils.compute_best_slices on our subsets.
    # This generates the parquet files with 'best_flair_index'.
    print("\n--- Preprocessing Subsets ---")
    train_processed = utils.compute_best_slices(
        train_subset_df, cache_name="train_subset", load_cached_data=False
    )
    val_processed = utils.compute_best_slices(
        val_subset_df, cache_name="val_subset", load_cached_data=False
    )
    test_processed = utils.compute_best_slices(
        test_subset_df, cache_name="test_subset", load_cached_data=False
    )

    # Verify processing added columns
    assert "best_flair_index" in train_processed.columns
    assert "num_flair_slices" in train_processed.columns
    print("Preprocessing verification passed.")

    # 4. Dataset & DataLoader
    print("\n--- Initializing Datasets & Loaders ---")
    batch_size = 4

    # Initialize Datasets
    train_ds = dataset.BraTSDataset(
        train_processed, phase="train", transform=dataset.get_transforms("train")
    )
    val_ds = dataset.BraTSDataset(val_processed, phase="val", transform=None)
    test_ds = dataset.BraTSDataset(test_processed, phase="test", transform=None)

    # Verify Dataset Output Shapes
    # Expected: ROI (12, 256, 256), Geo (12, 256, 256), Target (Scalar)
    sample_roi, sample_geo, sample_target = train_ds[0]
    print(f"Sample ROI Shape: {sample_roi.shape}")
    print(f"Sample Geo Shape: {sample_geo.shape}")
    print(f"Sample Target: {sample_target}")

    assert sample_roi.shape == (
        12,
        256,
        256,
    ), f"Unexpected ROI shape: {sample_roi.shape}"
    assert sample_geo.shape == (
        12,
        256,
        256,
    ), f"Unexpected Geo shape: {sample_geo.shape}"
    assert isinstance(sample_target, torch.Tensor), "Target should be a tensor"

    # Initialize Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print("DataLoaders initialized successfully.")

    # 5. Model Initialization
    print("\n--- Initializing Model ---")
    model = network.SiameseEfficientNet(
        backbone_name="efficientnet_b0", pretrained=True
    )
    model.to(device)

    # Verify Model Architecture (Input Stem)
    # The new stem should have 12 input channels
    stem_in_channels = model.backbone.conv_stem.in_channels
    print(f"Model Stem Input Channels: {stem_in_channels}")
    assert stem_in_channels == 12, "Model stem should accept 12 channels"

    # Verify Forward Pass
    dummy_roi = torch.randn(2, 12, 256, 256).to(device)
    dummy_geo = torch.randn(2, 12, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_roi, dummy_geo)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output (2, 1), got {output.shape}"
    print("Model forward pass verification passed.")

    # 6. Training Loop
    print("\n--- Running Training Loop (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_path = os.path.join(config.WORKING_DIR, "demo_best_model.pth")

    # Run training for 1 epoch with patience=1
    trained_model = engine.train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        patience=1,
        save_path=save_path,
    )

    # Verify model file creation
    if os.path.exists(save_path):
        print(f"Model checkpoint saved at: {save_path}")
    else:
        # It's possible validation didn't trigger save if AUC was -inf initially,
        # but train_loop logic sets best_auc = -inf, so first val should save unless error.
        print(
            "Note: Model might not be saved if validation failed to improve (unlikely for 1st epoch)."
        )

    # 7. Inference / Submission
    print("\n--- Generating Submission ---")
    # engine.generate_submission reads config.TEST_METADATA_PATH which we patched earlier
    engine.generate_submission(trained_model, test_loader, device)

    # Verify Submission File
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file created with shape: {sub_df.shape}")
        assert (
            len(sub_df) == subset_size
        ), f"Expected {subset_size} predictions, found {len(sub_df)}"
        assert "BraTS21ID" in sub_df.columns
        assert "MGMT_value" in sub_df.columns
        print("Submission format verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
