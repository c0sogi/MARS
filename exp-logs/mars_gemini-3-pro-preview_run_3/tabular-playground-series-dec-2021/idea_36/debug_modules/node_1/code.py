import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config, SEED, NUM_CLASSES
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data_processing import get_dataloaders
from library.model import DeepSupervisedNet
from library.train import train_one_epoch, validate, generate_submission


def create_mini_datasets(config):
    """
    Creates small subsets of the original data to speed up the demonstration.
    This avoids loading the full 3GB+ dataset into memory.
    """
    print("Creating mini datasets for rapid demonstration...")

    # Define mini paths in the working directory
    mini_train_path = os.path.join(config.working_dir, "train.parquet")
    mini_val_path = os.path.join(config.working_dir, "val.parquet")
    mini_test_path = os.path.join(config.working_dir, "test.parquet")

    # Load small chunks of original data (using original paths from default config)
    # We only take 500 rows for training and 200 for val/test
    df_train = pd.read_parquet(config.train_path).head(500)
    df_val = pd.read_parquet(config.val_path).head(200)
    df_test = pd.read_parquet(config.test_path).head(200)

    # Save to the new working directory locations
    df_train.to_parquet(mini_train_path, index=False)
    df_val.to_parquet(mini_val_path, index=False)
    df_test.to_parquet(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def demo_pipeline():
    # 1. Setup Configuration
    print("\n=== 1. Configuration Setup ===")
    # Initialize config in debug mode
    config = Config(debug=True)

    # Set a specific working directory for this demo to isolate outputs
    config.working_dir = "./working/demo_execution"
    os.makedirs(config.working_dir, exist_ok=True)

    # Create mini datasets to ensure the script runs quickly
    mini_train, mini_val, mini_test = create_mini_datasets(config)

    # Override config paths to point to the mini datasets
    config.train_path = mini_train
    config.val_path = mini_val
    config.test_path = mini_test

    # Set output path for the demo submission
    config.submission_path = os.path.join(config.working_dir, "submission_demo.csv")

    # Optimize hyperparameters for speed
    config.batch_size = 16
    config.epochs = 1
    config.dcn_layers = 1  # Reduce depth for speed
    config.resnet_blocks = 1  # Reduce depth for speed
    config.num_workers = 0  # Avoid multiprocessing overhead for tiny data

    print(f"Configured working directory: {config.working_dir}")
    print(f"Batch size: {config.batch_size}")

    # 2. Data Processing & Loading
    print("\n=== 2. Data Processing & Loading ===")

    # Clear any existing cache in the demo directory to force reprocessing of mini datasets
    cache_files = [
        "train_X.npy",
        "train_y.npy",
        "val_X.npy",
        "val_y.npy",
        "test_X.npy",
        "test_ids.npy",
    ]
    for f in cache_files:
        p = os.path.join(config.working_dir, f)
        if os.path.exists(p):
            os.remove(p)

    # Load data using the library function
    # load_cached_data=False forces the processor to read our new mini parquets
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        config, load_cached_data=False
    )

    print(f"Input dimension detected: {input_dim}")
    print(f"Train batches: {len(train_loader)}")

    # Verification
    # Original features (54) + Engineered features (Aspect_Sin, Aspect_Cos, Euclidean, Abs_Hydro, Mean_Dist)
    # Expected input_dim >= 54. Actually 54 + 5 = 59.
    assert input_dim >= 54, f"Expected at least 54 features, got {input_dim}"

    # Verify batch structure
    sample_X, sample_y = next(iter(train_loader))
    assert sample_X.shape == (
        config.batch_size,
        input_dim,
    ), "Feature shape mismatch in loader"
    assert sample_y.shape == (config.batch_size,), "Target shape mismatch in loader"
    print("Data loading verification passed.")

    # 3. Model Initialization
    print("\n=== 3. Model Initialization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DeepSupervisedNet(input_dim, NUM_CLASSES, config).to(device)

    # Verification: Forward pass
    dummy_input = torch.randn(2, input_dim).to(device)
    prim_logits, aux_logits = model(dummy_input)

    # Check output shapes
    assert prim_logits.shape == (
        2,
        NUM_CLASSES,
    ), f"Primary logits shape mismatch: {prim_logits.shape}"
    # Aux logits might be None if the aux head isn't attached at the specific block,
    # but based on config resnet_blocks=1 and aux_attach_idx=2 in model.py,
    # aux_logits might be None if the loop doesn't reach index 2.
    # However, let's check if it runs without error.
    # In the provided model.py: self.aux_attach_idx = 2.
    # If config.resnet_blocks = 1, the loop runs for i=0. i never equals 2.
    # So aux_logits will be None. This is expected behavior for shallow networks.
    if config.resnet_blocks > 2:
        assert aux_logits.shape == (2, NUM_CLASSES), "Aux logits shape mismatch"
    else:
        assert (
            aux_logits is None
        ), "Aux logits should be None for shallow network (blocks < 3)"

    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n=== 4. Training Loop Demonstration ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    loss, acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, config
    )
    print(f"Train Step - Loss: {loss:.4f}, Acc: {acc:.4f}")

    assert not np.isnan(loss), "Training loss is NaN"
    assert 0 <= acc <= 1, "Training accuracy out of bounds"

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Validation Step - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 5. Checkpointing Demonstration
    print("\n=== 5. Checkpointing Demonstration ===")
    ckpt_path = os.path.join(config.working_dir, "best_model.pth")

    # Save checkpoint
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "best_acc": val_acc,
    }
    save_checkpoint(state, ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    # Load checkpoint
    # We create a new model instance to ensure weights are actually loaded
    model_loaded = DeepSupervisedNet(input_dim, NUM_CLASSES, config).to(device)
    loaded_checkpoint = load_checkpoint(ckpt_path, model_loaded, device)

    assert "state_dict" in loaded_checkpoint, "Loaded checkpoint missing state_dict"
    print("Checkpoint save/load verification passed.")

    # 6. Submission Generation
    print("\n=== 6. Submission Generation ===")
    # Use the loaded model to generate submission
    generate_submission(model_loaded, test_loader, test_ids, device, config)

    assert os.path.exists(config.submission_path), "Submission file not created"

    # Verify submission content
    sub_df = pd.read_csv(config.submission_path)
    assert list(sub_df.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"
    assert len(sub_df) == len(
        test_ids
    ), f"Submission row count mismatch: {len(sub_df)} vs {len(test_ids)}"
    assert (
        sub_df["Cover_Type"].dtype == np.int64
    ), "Prediction column should be integers"

    print("Submission generation verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    seed_everything(SEED)
    demo_pipeline()
