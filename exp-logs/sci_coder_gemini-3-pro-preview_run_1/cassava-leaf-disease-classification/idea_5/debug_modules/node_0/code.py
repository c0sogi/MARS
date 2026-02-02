import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, Mixup
from library.model import CassavaModel
from library.engine import train_one_epoch, validate, generate_submission


def run_demonstration():
    print("==== Starting Cassava Leaf Disease Classification Demo ====")

    # 1. Configuration Setup
    # We use debug=True to trigger internal logic for subsets, but we also manually
    # override specific params to ensure it fits the 'fast demo' requirement perfectly.
    config = Config(debug=True)

    # Override paths to isolate this execution
    config.working_dir = "./working/demo_execution"
    config.output_dir = config.working_dir
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    # Ensure directories exist
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    # Optimization for speed
    config.epochs_stage1 = 1
    config.batch_size = 8
    config.input_size_stage1 = 224  # Reduced size for faster processing
    config.data_subset_fraction = 0.02  # Use 2% of data for quick iteration
    config.num_workers = 2

    # Disable pretrained weights to avoid downloading overhead/errors
    # In a real run, this would be True.
    use_pretrained = False

    print(f"Working Directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # 2. Seeding
    print("\n[Step 1] Setting Seeds...")
    seed_everything(config.seed)

    # 3. Data Loading
    print("\n[Step 2] Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, stage=1)

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"  Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        assert images.shape[0] == config.batch_size, "Batch size mismatch"
        assert images.shape[1] == 3, "Channel count mismatch"
        assert labels.shape[0] == config.batch_size, "Label batch size mismatch"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 4. Mixup Augmentation Verification
    print("\n[Step 3] Verifying Mixup Logic...")
    mixup_fn = Mixup(config)
    # Force probability to 1.0 to ensure it runs
    mixup_fn.mixup_prob = 1.0

    mixed_images, mixed_targets = mixup_fn(
        images.to(config.device), labels.to(config.device)
    )

    print(f"  Mixed Images: {mixed_images.shape}, Mixed Targets: {mixed_targets.shape}")
    assert mixed_images.shape == images.shape
    # Targets should be (Batch, Num_Classes) due to one-hot encoding in Mixup
    assert mixed_targets.shape == (config.batch_size, config.num_classes)
    assert mixed_targets.dtype == torch.float32 or mixed_targets.dtype == torch.float16

    # 5. Model Initialization
    print("\n[Step 4] Initializing Model...")
    model = CassavaModel(config, pretrained=use_pretrained)
    model.to(config.device)

    # Verify Forward Pass with Dummy Data
    dummy_input = torch.randn(
        2, 3, config.input_size_stage1, config.input_size_stage1
    ).to(config.device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"  Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, config.num_classes)

    # 6. Training Loop
    print("\n[Step 5] Running Training (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_stage1)

    # Ensure mixup is applied during training function call
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=config.device,
        config=config,
        mixup_fn=mixup_fn,
    )

    print(f"  Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # 7. Validation Loop
    print("\n[Step 6] Running Validation...")
    val_acc, val_loss = validate(model, val_loader, config.device, config)

    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    assert 0.0 <= val_acc <= 1.0, "Accuracy out of bounds"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 8. Checkpointing
    print("\n[Step 7] Verifying Checkpoint Save/Load...")
    ckpt_path = os.path.join(config.checkpoint_dir, "demo_model.pth")

    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
        },
        ckpt_path,
    )

    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    # Reload to verify
    loaded_ckpt = load_checkpoint(ckpt_path, model, optimizer)
    assert "model_state_dict" in loaded_ckpt
    print("  Checkpoint saved and reloaded successfully.")

    # 9. Inference and Submission
    print("\n[Step 8] Generating Submission...")
    generate_submission(model, test_loader, config.device, config)

    assert os.path.exists(config.submission_path), "Submission file not found"

    df_sub = pd.read_csv(config.submission_path)
    print(f"  Submission Rows: {len(df_sub)}")
    print("  Head:")
    print(df_sub.head(2))

    assert list(df_sub.columns) == ["image_id", "label"], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"
    # Check if labels are integers
    assert pd.api.types.is_integer_dtype(
        df_sub["label"]
    ), "Label column should be integer"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demonstration()
