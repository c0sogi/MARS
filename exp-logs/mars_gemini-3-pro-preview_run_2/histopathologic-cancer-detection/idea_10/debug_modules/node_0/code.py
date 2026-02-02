import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

# Import provided library modules
import library.config as config_module
from library.config import Config
import library.data as data
import library.model as model
import library.training as training
import library.inference as inference
import library.utils as utils


def run_demo():
    print("=== Starting Histopathology Tumor Detection Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override
    # ---------------------------------------------------------
    # We override the default configuration to run a fast, isolated demo.

    # Define a separate working directory for this demo
    demo_working_dir = "./working/demo_execution"

    # Clean up previous demo runs if they exist
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    # Update Config paths (since they are static class attributes)
    Config.working_dir = demo_working_dir
    Config.cache_dir = os.path.join(demo_working_dir, "cache")
    Config.checkpoints_dir = os.path.join(demo_working_dir, "checkpoints")
    Config.submission_dir = os.path.join(demo_working_dir, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Optimize hyperparameters for speed
    Config.debug = True
    Config.debug_sample_size = 20  # Use only 20 samples for training/inference
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 4  # Small batch size
    Config.n_folds = 2  # Setup for 2 folds, but we will only run Fold 0
    Config.num_workers = 0  # Disable multiprocessing for simple script execution

    # Create directories and set seeds
    Config.setup()

    print(f"Configuration updated. Working directory: {Config.working_dir}")

    # ---------------------------------------------------------
    # 2. Data Loading (Mocked for Speed)
    # ---------------------------------------------------------
    print("\n--- Step 1: Loading Data (Mocked) ---")

    # We patch pandas.read_csv to read only the first 50 rows of the metadata.
    # This prevents the data loader from reading all 140k images from disk.
    original_read_csv = pd.read_csv

    def mocked_read_csv(filepath, *args, **kwargs):
        # Call original but slice the result
        df = original_read_csv(filepath, *args, **kwargs)
        return df.head(50)

    with patch("pandas.read_csv", side_effect=mocked_read_csv):
        # Load data to memory (load_cached_data=False to force execution of loading logic)
        train_images, train_labels, test_images, test_ids = data.load_data_to_memory(
            load_cached_data=False
        )

    # Verification
    # We expect 100 train images (50 from train.csv + 50 from val.csv) and 50 test images
    print(f"Loaded Train Images Shape: {train_images.shape}")
    print(f"Loaded Test Images Shape: {test_images.shape}")

    assert (
        len(train_images) == 100
    ), "Expected 100 training images (mocked 50 train + 50 val)."
    assert len(test_images) == 50, "Expected 50 test images (mocked)."
    assert train_images.dtype == np.uint8, "Images should be uint8."

    # ---------------------------------------------------------
    # 3. Model Logic Verification
    # ---------------------------------------------------------
    print("\n--- Step 2: Verifying Model Architecture ---")

    # Instantiate Model
    net = model.get_model()
    net.to(Config.device)

    # Create a dummy input batch: (Batch=4, Channels=3, Height=64, Width=64)
    # Note: 64x64 is the crop size defined in Config.image_crop_size
    dummy_input = torch.randn(4, 3, 64, 64).to(Config.device)

    # A. Verify Training Mode (Multi-Sample Dropout)
    net.train()
    out_train = net(dummy_input)
    print(f"Model Output (Train): {out_train.shape}")
    # Expect: (Batch, MSD_Count, 1)
    assert out_train.shape == (
        4,
        Config.multi_sample_dropout_count,
        1,
    ), f"Incorrect train output shape. Expected (4, {Config.multi_sample_dropout_count}, 1)."

    # B. Verify Evaluation Mode (Ensemble Average)
    net.eval()
    out_eval = net(dummy_input)
    print(f"Model Output (Eval): {out_eval.shape}")
    # Expect: (Batch, 1)
    assert out_eval.shape == (4, 1), "Incorrect eval output shape. Expected (4, 1)."

    # C. Verify Loss Function
    loss_fn = model.get_loss_fn()
    dummy_target = (
        torch.tensor([0, 1, 0, 1], dtype=torch.float32).to(Config.device).view(-1, 1)
    )

    # Calculate loss using training output (MSD logits)
    loss = loss_fn(out_train, dummy_target)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."

    # ---------------------------------------------------------
    # 4. Training Loop Verification
    # ---------------------------------------------------------
    print("\n--- Step 3: Verifying Training Loop (Fold 0) ---")

    # Get DataLoaders for Fold 0
    # Config.debug=True will slice these to Config.debug_sample_size (20)
    train_loader, val_loader = data.get_fold_dataloaders(0, train_images, train_labels)

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Initialize Trainer
    trainer = training.Trainer(0, train_loader, val_loader)

    # Run Training (1 Epoch)
    trainer.fit()

    # Check if checkpoint was saved
    best_ckpt_path = os.path.join(Config.checkpoints_dir, "best_model_fold_0.pth")
    last_ckpt_path = os.path.join(Config.checkpoints_dir, "last_model_fold_0.pth")

    assert os.path.exists(best_ckpt_path), "Best model checkpoint not found."
    assert os.path.exists(last_ckpt_path), "Last model checkpoint not found."
    print("Training finished and checkpoints verified.")

    # ---------------------------------------------------------
    # 5. Inference Verification
    # ---------------------------------------------------------
    print("\n--- Step 4: Verifying Inference Pipeline ---")

    # Get Test DataLoader
    # Config.debug=True will slice test set to 20 samples
    test_loader = data.get_test_dataloader(test_images)

    # Run Inference using the model trained in Step 4
    preds = inference.inference_single_fold(0, test_loader, Config.device)

    print(f"Predictions Shape: {preds.shape}")

    # Verify Shape
    expected_samples = Config.debug_sample_size
    assert (
        len(preds) == expected_samples
    ), f"Expected {expected_samples} predictions, got {len(preds)}."

    # Verify Value Range
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions contain values outside [0, 1]."

    # Verify TTA Helper
    print("Verifying TTA helper...")
    images_batch = next(iter(test_loader))
    tta_out = inference.predict_batch_with_tta(net, images_batch, Config.device)
    assert tta_out.shape == (images_batch.size(0), 1), "TTA output shape mismatch."

    print("Inference verification successful.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
