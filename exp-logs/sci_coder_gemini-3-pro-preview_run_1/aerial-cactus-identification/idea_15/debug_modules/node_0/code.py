import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_manager import (
    load_all_train_data,
    load_test_data,
    get_file_size_stats,
    normalize_file_sizes,
    get_transforms,
    CactusDataset,
)
from library.models import RepVGG_FiLM, ResNet_FiLM, NeXt_FiLM, reparameterize_model
from library.training_engine import train_one_epoch, validate, SWAHandler
from library.inference_engine import predict_loader
from library.stacking import fit_meta_learner, predict_ensemble, save_submission


def run_demonstration():
    print("=" * 40)
    print(" STARTING PIPELINE DEMONSTRATION")
    print("=" * 40)

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration...")

    # Override Config for a fast demonstration run
    Config.DEBUG = True  # Forces data loaders to return a small subset (256 samples)
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.SWA_START_EPOCH = 0  # Trigger SWA immediately for demo

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print("Debug Mode: Enabled (using data subsets)")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2] Loading and processing data...")

    # Load data (Config.DEBUG=True ensures we only load a small slice)
    # We set load_cached_data=False to demonstrate the raw loading logic
    train_imgs, train_labels, train_ids, train_fsizes = load_all_train_data(
        load_cached_data=False
    )
    test_imgs, test_labels, test_ids, test_fsizes = load_test_data(
        load_cached_data=False
    )

    print(f"Train Subset Shape: {train_imgs.shape}")
    print(f"Test Subset Shape:  {test_imgs.shape}")

    # Validate data integrity
    assert len(train_imgs) == len(train_labels) == len(train_fsizes)
    assert train_imgs.dtype == np.uint8
    assert train_imgs.shape[1:] == (32, 32, 3)

    # Calculate and apply normalization for file sizes
    fsize_mean, fsize_std = get_file_size_stats(train_fsizes)
    train_fsizes_norm = normalize_file_sizes(train_fsizes, fsize_mean, fsize_std)
    test_fsizes_norm = normalize_file_sizes(test_fsizes, fsize_mean, fsize_std)

    # Create Datasets and DataLoaders
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # Split the subset into train/val for demonstration
    split_idx = int(len(train_imgs) * 0.8)

    ds_train = CactusDataset(
        train_imgs[:split_idx],
        train_labels[:split_idx],
        train_fsizes_norm[:split_idx],
        train_ids[:split_idx],
        transform=train_transform,
    )

    ds_val = CactusDataset(
        train_imgs[split_idx:],
        train_labels[split_idx:],
        train_fsizes_norm[split_idx:],
        train_ids[split_idx:],
        transform=val_transform,
    )

    loader_train = DataLoader(
        ds_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    loader_val = DataLoader(
        ds_val, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify a single batch
    batch_imgs, batch_labels, batch_fsizes, batch_ids = next(iter(loader_train))
    print(
        f"Batch Shapes -> Img: {batch_imgs.shape}, Labels: {batch_labels.shape}, Fsizes: {batch_fsizes.shape}"
    )
    assert batch_imgs.shape == (Config.BATCH_SIZE, 3, 32, 32)

    # -------------------------------------------------------------------------
    # 3. Model Logic Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architectures...")

    # List of model classes to verify
    model_classes = [RepVGG_FiLM, ResNet_FiLM, NeXt_FiLM]

    for model_cls in model_classes:
        print(f"  Testing {model_cls.__name__}...")
        model = model_cls(num_classes=1).to(device)

        # Run a forward pass
        dummy_img = batch_imgs.to(device)
        dummy_fsize = batch_fsizes.to(device)

        output = model(dummy_img, dummy_fsize)

        # Check output structure
        assert "logits" in output
        assert "aux_logits" in output
        assert output["logits"].shape == (Config.BATCH_SIZE, 1)

        # Special check for RepVGG reparameterization
        if isinstance(model, RepVGG_FiLM):
            print("    Verifying RepVGG reparameterization...")
            model.eval()
            with torch.no_grad():
                out_before = model(dummy_img, dummy_fsize)["logits"]

            # Switch to deploy mode (fuses Conv+BN)
            reparameterize_model(model)
            assert model.deploy is True

            with torch.no_grad():
                out_after = model(dummy_img, dummy_fsize)["logits"]

            # Check that outputs are numerically consistent
            diff = (out_before - out_after).abs().max().item()
            print(f"    Difference after fusion: {diff:.6f}")
            assert (
                diff < 1e-4
            ), "RepVGG fusion resulted in significant output deviation."

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop...")

    # Initialize model (NeXt_FiLM)
    model = NeXt_FiLM(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # SWA Handler
    swa_handler = SWAHandler(model, optimizer, device)

    # Train for one epoch
    print("  Training for 1 epoch...")
    train_loss = train_one_epoch(
        model, loader_train, optimizer, criterion, device, epoch=0
    )
    print(f"  Train Loss: {train_loss:.4f}")

    # Trigger SWA update (mocking epoch condition)
    swa_active = swa_handler.step(epoch=Config.SWA_START_EPOCH)
    print(f"  SWA Update Triggered: {swa_active}")

    # Validate
    val_loss, val_auc = validate(model, loader_val, criterion, device)
    print(f"  Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Finalize SWA (Update BN stats)
    print("  Finalizing SWA model...")
    swa_model = swa_handler.finalize(loader_train)
    assert isinstance(swa_model, torch.nn.Module)

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference with TTA...")

    # Create test loader
    ds_test = CactusDataset(
        test_imgs,
        test_labels,  # Dummy labels
        test_fsizes_norm,
        test_ids,
        transform=val_transform,
    )
    loader_test = DataLoader(
        ds_test, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Predict using the trained model
    preds, ids = predict_loader(model, loader_test, device)

    print(f"  Predictions generated: {len(preds)}")
    assert len(preds) == len(test_imgs)
    assert len(ids) == len(test_imgs)
    assert preds.min() >= 0.0 and preds.max() <= 1.0

    # -------------------------------------------------------------------------
    # 6. Stacking Ensemble Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Stacking Ensemble...")

    # Simulate OOF predictions for 2 models
    num_samples = 100
    y_true = np.random.randint(0, 2, num_samples)
    # Ensure both classes exist
    y_true[:5] = 0
    y_true[5:10] = 1

    oof_preds = {
        "RepVGG": np.random.rand(num_samples),
        "NeXt": np.random.rand(num_samples),
    }

    # Fit meta-learner
    print("  Fitting meta-learner...")
    meta_model = fit_meta_learner(
        oof_preds, y_true, save_path="./working/meta_model.joblib"
    )

    # Simulate Test predictions
    test_preds_dict = {"RepVGG": np.random.rand(10), "NeXt": np.random.rand(10)}

    # Predict with ensemble
    final_probs = predict_ensemble(meta_model, test_preds_dict)
    print(f"  Ensemble predictions shape: {final_probs.shape}")

    # Save submission
    dummy_ids = [f"test_{i}.jpg" for i in range(10)]
    save_submission(dummy_ids, final_probs, output_path="./working/demo_submission.csv")

    print("\n" + "=" * 40)
    print(" DEMONSTRATION COMPLETE SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demonstration()
