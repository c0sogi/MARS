import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, weighted_auc, AverageMeter
from library.srm_filters import get_srm_layer
from library.dataset import (
    StegoDataset,
    TestDataset,
    get_transforms,
    process_grouped_metadata,
)
from library.model import ResV2GeM
from library.engine import train_one_epoch, validate, predict_tta


def run_demo():
    print("=== Starting Library Verification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Demo Configuration...")

    # Override Config paths and settings to use a temporary working directory
    # and run quickly without downloading heavy weights.
    Config.working_dir = "./working/demo_run"
    Config.checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")
    Config.predictions_dir = os.path.join(Config.working_dir, "predictions")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    Config.epochs = 1
    Config.batch_size = 4
    Config.pretrained = False  # Disable downloading weights for speed
    Config.unique_content_sampling = True

    # Create necessary directories
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.predictions_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)
    print("    Configuration updated and directories created.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Weighted AUC
    # Case A: Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    score_perfect = weighted_auc(y_true, y_pred_perfect)
    print(f"    Weighted AUC (Perfect): {score_perfect:.4f}")
    assert score_perfect == 1.0, "Metric failed on perfect predictions."

    # Case B: Worst predictions
    y_pred_worst = np.array([0.9, 0.8, 0.2, 0.1])
    score_worst = weighted_auc(y_true, y_pred_worst)
    print(f"    Weighted AUC (Worst):   {score_worst:.4f}")
    assert score_worst == 0.0, "Metric failed on worst predictions."

    # Case C: Random/Constant (should be 0.5)
    y_pred_const = np.array([0.5, 0.5, 0.5, 0.5])
    score_const = weighted_auc(y_true, y_pred_const)
    print(f"    Weighted AUC (Const):   {score_const:.4f}")
    assert score_const == 0.5, "Metric failed on constant predictions."

    # -------------------------------------------------------------------------
    # 3. Verify SRM Filters
    # -------------------------------------------------------------------------
    print("\n[3] Verifying SRM Layer...")
    srm_layer = get_srm_layer()

    # Create dummy input: (Batch=2, Channels=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    srm_output = srm_layer(dummy_input)

    print(f"    Input Shape:  {dummy_input.shape}")
    print(f"    Output Shape: {srm_output.shape}")

    # SRM layer should produce 30 channels (filters)
    assert srm_output.shape == (2, 30, 256, 256), "SRM output shape mismatch."
    assert not srm_layer.weight.requires_grad, "SRM weights should be frozen."
    print("    SRM Layer verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Dataset & Grouping Logic
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset and Metadata Grouping...")

    # Create a small subset of the training metadata
    full_train_df = pd.read_csv(Config.train_csv)

    # Select 5 unique image IDs to form a mini dataset (5 IDs * 4 variants = 20 images)
    unique_ids = full_train_df["image_id"].unique()[:5]
    subset_train_df = full_train_df[full_train_df["image_id"].isin(unique_ids)].copy()

    demo_train_csv = os.path.join(Config.working_dir, "demo_train.csv")
    subset_train_df.to_csv(demo_train_csv, index=False)
    print(f"    Created demo training CSV with {len(subset_train_df)} rows.")

    # Test Grouping Function directly
    grouped_df = process_grouped_metadata(
        demo_train_csv, Config.cache_dir, load_cached_data=False
    )
    print(f"    Grouped DataFrame Shape: {grouped_df.shape}")
    # Should have 5 rows (one per image_id) and columns for Cover + 3 Algos
    assert len(grouped_df) == 5, "Grouping failed: Incorrect number of rows."
    assert all(
        col in grouped_df.columns for col in ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    ), "Missing algo columns."

    # Test StegoDataset (Train Mode - Unique Content Sampling)
    train_dataset = StegoDataset(
        csv_path=demo_train_csv,
        root_dir=Config.input_dir,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    print(f"    Train Dataset Length: {len(train_dataset)}")
    assert (
        len(train_dataset) == 5
    ), "Train dataset length mismatch (expected grouped length)."

    # Fetch one sample
    img, label = train_dataset[0]
    print(f"    Sample Image Shape: {img.shape}, Label: {label}")
    assert img.shape == (3, 512, 512), "Image tensor shape mismatch."
    assert isinstance(label, torch.Tensor), "Label is not a tensor."

    # Test StegoDataset (Val Mode - Flat)
    val_dataset = StegoDataset(
        csv_path=demo_train_csv,  # Using same subset for demo
        root_dir=Config.input_dir,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=False,
    )
    print(f"    Val Dataset Length: {len(val_dataset)}")
    assert len(val_dataset) == 20, "Val dataset length mismatch (expected flat length)."

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")
    device = Config.device
    model = ResV2GeM(config=Config, pretrained=False)
    model.to(device)

    # Forward pass with dummy data
    dummy_batch = torch.randn(2, 3, 512, 512).to(device)
    logits = model(dummy_batch)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (2, 1), "Model output shape mismatch."
    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 6. Verify Engine (Training & Inference Loop)
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Engine (Train/Val/Predict)...")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # A. Train One Epoch
    print("    Running training step...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # B. Validate
    print("    Running validation step...")
    val_auc, val_loss = validate(model, val_loader, device)
    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert 0.0 <= val_auc <= 1.0, "Validation AUC out of range."

    # C. Predict (TTA)
    print("    Running inference step...")
    # Create a small test subset
    test_files = sorted(os.listdir(os.path.join(Config.input_dir, "Test")))[:5]
    test_df = pd.DataFrame(
        {
            "image_id": test_files,
            "file_path": [os.path.join("Test", f) for f in test_files],
        }
    )
    demo_test_csv = os.path.join(Config.working_dir, "demo_test.csv")
    test_df.to_csv(demo_test_csv, index=False)

    test_dataset = TestDataset(
        csv_path=demo_test_csv,
        root_dir=Config.input_dir,
        transform=get_transforms("val"),  # No augmentation for base test
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, num_workers=0)

    # Run TTA
    preds_df = predict_tta(model, test_loader, device)

    print("    Predictions Head:")
    print(preds_df.head())

    assert len(preds_df) == 5, "Prediction count mismatch."
    assert (
        "Id" in preds_df.columns and "Label" in preds_df.columns
    ), "Submission format incorrect."

    # Save submission
    preds_df.to_csv(Config.submission_path, index=False)
    print(f"    Submission saved to {Config.submission_path}")

    print("\n=== All Verification Steps Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
