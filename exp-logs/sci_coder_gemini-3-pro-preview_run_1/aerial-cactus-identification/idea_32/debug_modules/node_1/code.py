import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
import logging
from torch.utils.data import DataLoader

# Import library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.models as models
import library.engine as engine
import library.inference as inference
import library.stacking as stacking

# Configure logging to suppress verbose output from libraries if needed
logging.getLogger("engine").setLevel(logging.WARNING)
logging.getLogger("stacking").setLevel(logging.INFO)


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setup and Configuration")
    config.seed_everything(config.SEED)
    config.setup_directories()

    # Device configuration
    device = config.DEVICE
    print(f"Using device: {device}")

    # Monkey-patch NUM_FOLDS to 2 for speed in this demo
    original_num_folds = config.NUM_FOLDS
    config.NUM_FOLDS = 2
    stacking.NUM_FOLDS = 2
    print(
        f"Reduced NUM_FOLDS from {original_num_folds} to {config.NUM_FOLDS} for demo speed."
    )

    # 2. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[2] Dataset and DataLoader")

    # Load training metadata
    print("Loading training data...")
    train_ids, train_imgs, train_lbls = dataset.load_and_cache_data(
        config.TRAIN_META_PATH, "demo_train", load_cached_data=False
    )

    # Verify data shapes
    print(f"Loaded {len(train_ids)} training samples.")
    assert len(train_ids) == len(train_imgs) == len(train_lbls)
    assert train_imgs.shape[1:] == (32, 32, 3)

    # Create Dataset
    # Use a small subset for the demo loop
    subset_indices = np.arange(100)
    train_ds = dataset.CactusDataset(
        train_ids[subset_indices],
        train_imgs[subset_indices],
        train_lbls[subset_indices],
        transform=dataset.get_transforms("train"),
    )

    # Test __getitem__
    sample = train_ds[0]
    assert "image" in sample and "target" in sample and "id" in sample
    assert sample["image"].shape == (3, 32, 32)
    print("Dataset __getitem__ verification passed.")

    # Create DataLoader with Mixup Collate
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.mixup_collate_fn,
        num_workers=0,
    )

    # Test Batch
    batch = next(iter(train_loader))
    assert "lam" in batch
    assert batch["image"].shape == (16, 3, 32, 32)
    assert batch["target_a"].shape == (16,)
    print("DataLoader with Mixup collate verification passed.")

    # 3. Model Architectures
    # -------------------------------------------------------------------------
    print("\n[3] Model Architectures")

    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    for arch in config.MODEL_ARCHITECTURES:
        print(f"Testing architecture: {arch}")
        model = models.ModelFactory.get_model(arch).to(device)

        # Test Training Forward Pass (Expect tuple: main, aux)
        model.train()
        out = model(dummy_input)
        assert isinstance(out, tuple), f"{arch} should return tuple in train mode"
        main_out, aux_out = out
        assert main_out.shape == (4, 1)
        assert aux_out.shape == (4, 1)

        # Test Eval Forward Pass (Expect tensor)
        model.eval()
        out = model(dummy_input)
        assert isinstance(
            out, torch.Tensor
        ), f"{arch} should return tensor in eval mode"
        assert out.shape == (4, 1)

        # Test Reparameterization for RepVGG
        if arch == "RepVGG":
            print("  Verifying RepVGG reparameterization...")
            # Check if attributes exist before reparam
            assert hasattr(model.stage1[0], "rbr_dense")
            model.reparameterize()
            # Check if attributes are removed after reparam
            assert not hasattr(model.stage1[0], "rbr_dense")
            # Verify forward pass still works
            out_reparam = model(dummy_input)
            assert out_reparam.shape == (4, 1)

    print("All model architectures verified.")

    # 4. Training Engine
    # -------------------------------------------------------------------------
    print("\n[4] Training Engine")

    # Setup a simple training loop for verification
    model = models.ModelFactory.get_model("NeXt").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print("Running train_one_epoch...")
    loss = engine.train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"Epoch Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # Validation
    val_ds = dataset.CactusDataset(
        train_ids[100:150],
        train_imgs[100:150],
        train_lbls[100:150],
        transform=dataset.get_transforms("val"),
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    print("Running validate...")
    val_loss, val_auc = engine.validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # SWA Handler
    print("Testing SWA Handler...")
    swa_handler = engine.SWAHandler(
        model, optimizer, swa_start_epoch=0, swa_lr=1e-4, device=device
    )
    swa_handler.on_epoch_end(epoch=1)
    assert swa_handler.active
    swa_handler.update_bn(train_loader)
    print("SWA Handler verification passed.")

    # Early Stopping
    print("Testing Early Stopping...")
    es = engine.EarlyStopping(patience=2, mode="max")
    assert es(0.5) is True  # New best
    assert es(0.4) is False  # Worse
    assert es(0.4) is False  # Worse
    assert es.early_stop is True
    print("Early Stopping verification passed.")

    # 5. Inference and Checkpointing
    # -------------------------------------------------------------------------
    print("\n[5] Inference and Checkpointing")

    # Save a dummy checkpoint to use for stacking demo
    checkpoint_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "best_score": val_auc,
    }

    # We need to populate the checkpoint directory with dummy files for the stacking script
    # Stacking expects: {model_name}_fold{fold}.pth
    print("Generating dummy checkpoints for stacking...")
    for arch in config.MODEL_ARCHITECTURES:
        # Create a fresh model instance for the architecture to match state dict keys
        temp_model = models.ModelFactory.get_model(arch).to(device)
        temp_state = {
            "model_state_dict": temp_model.state_dict(),
            "optimizer_state_dict": {},  # Empty is fine for inference loading
        }
        for fold in range(config.NUM_FOLDS):
            filename = f"{arch}_fold{fold}.pth"
            utils.save_checkpoint(temp_state, filename)

    print(f"Saved dummy checkpoints to {config.CHECKPOINT_DIR}")

    # Test TTA Inference
    print("Testing TTA Inference...")
    ids, raw_preds, targets = inference.predict_with_tta(model, val_loader, device)
    assert raw_preds.shape == (len(val_ds), 4)  # 4 TTA views
    assert len(ids) == len(val_ds)

    # Test Prediction Processing
    df_preds = inference._process_predictions(ids, raw_preds, targets)
    assert "pred_mean" in df_preds.columns
    assert "pred_std" in df_preds.columns
    print("Inference verification passed.")

    # 6. Stacking Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Stacking Pipeline")

    # We will run the stacking pipeline.
    # It will:
    # 1. Load data (we use cache=False to force reload/recompute for demo)
    # 2. Iterate folds (we reduced to 2)
    # 3. Load the dummy checkpoints we created
    # 4. Generate OOF and Test predictions
    # 5. Train the meta-learner

    # Note: This might take a minute as it runs inference on the full dataset using dummy models.
    # To speed it up further, we can mock the data loading inside stacking, but let's try running it
    # as the dataset is small (32x32 images).

    try:
        stacking.train_stacking_model(load_cached_data=False)

        # Verify outputs
        assert os.path.exists(config.SUBMISSION_FILE)
        submission_df = pd.read_csv(config.SUBMISSION_FILE)
        print(f"Submission file created with {len(submission_df)} rows.")
        assert "id" in submission_df.columns and "has_cactus" in submission_df.columns

        meta_model_path = os.path.join(config.BASE_OUTPUT_DIR, "meta_model.joblib")
        assert os.path.exists(meta_model_path)
        print("Meta-model saved successfully.")

    except Exception as e:
        print(f"Stacking pipeline failed: {e}")
        raise e

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
