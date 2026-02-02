import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_loaders, prepare_folds, get_test_loader
from library.modeling import CassavaClassifier, get_optimizer_params
from library.loss import SoftTargetCrossEntropy
from library.engine import train_one_epoch, validate
from library.inference import run_inference


def run_demo():
    print("--- Starting Cassava Leaf Disease Classification Demo ---")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")
    seed_everything(42)

    # Override Config defaults for a quick demonstration
    Config.debug = True  # Use a small subset of data (200 train, 100 val)
    Config.n_folds = 2  # Simulate 2 folds, but we will only run Fold 0
    Config.p1_epochs = 1  # Run only 1 epoch
    Config.p1_batch_size = 8  # Small batch size for speed
    Config.p1_image_size = 224  # Standard small image size
    Config.p2_image_size = 224  # Keep consistent for inference
    Config.model_names = ["resnet18"]  # Use a lightweight model supported by timm
    Config.output_dir = "./working/demo_output"  # Custom output dir for demo
    Config.num_workers = 2  # Reduce workers to minimize overhead
    Config.print_freq = 10  # Print logs more frequently

    # Clean up previous demo output if it exists
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    os.makedirs(Config.output_dir, exist_ok=True)

    print("Configuration updated: Debug Mode=True, Model=resnet18, Epochs=1")

    # ---------------------------------------------------------
    # 2. Dataset and DataLoader Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Prepare folds (creates folds.parquet in output_dir)
    df_folds = prepare_folds(load_cached_data=False)
    print(f"Folds DataFrame created. Shape: {df_folds.shape}")
    assert "fold" in df_folds.columns, "Folds DataFrame missing 'fold' column"

    # Instantiate DataLoaders for Fold 0
    train_loader, val_loader = get_loaders(
        fold=0,
        image_size=Config.p1_image_size,
        batch_size=Config.p1_batch_size,
        debug=Config.debug,
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches:   {len(val_loader)}")

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Sample Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.p1_batch_size,
        3,
        Config.p1_image_size,
        Config.p1_image_size,
    )
    assert labels.shape == (Config.p1_batch_size,)
    assert labels.max() < Config.num_classes

    # ---------------------------------------------------------
    # 3. Modeling Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Initialization...")
    device = Config.device

    # Initialize model
    # pretrained=False ensures we don't depend on internet access for this demo
    model = CassavaClassifier(model_name="resnet18", pretrained=False)
    model.to(device)

    # Perform a forward pass
    with torch.no_grad():
        logits = model(images.to(device))

    print(f"Model Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.p1_batch_size, Config.num_classes)

    # Verify Optimizer Parameter Grouping (LLRD)
    param_groups = get_optimizer_params(model, base_lr=1e-3, weight_decay=1e-4)
    print(f"Optimizer Parameter Groups: {len(param_groups)}")
    # ResNet usually has Stem + 4 Stages + Head = ~6 groups
    assert (
        len(param_groups) > 1
    ), "Layer-wise Learning Rate Decay failed to group parameters."

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")
    criterion = SoftTargetCrossEntropy()

    # Create dummy one-hot targets to simulate MixUp/CutMix targets
    targets_one_hot = F.one_hot(
        labels.to(device), num_classes=Config.num_classes
    ).float()

    loss = criterion(logits, targets_one_hot)
    print(f"Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss computation resulted in NaN."
    assert loss.item() > 0, "Loss should be positive."

    # ---------------------------------------------------------
    # 5. Training Loop Execution (Engine)
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(param_groups)

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloader=train_loader,
        device=device,
        epoch=0,
        accum_steps=1,
    )

    print(f"Training Completed. Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # Validate
    val_loss, val_acc = validate(model, val_loader, device)
    print(f"Validation Completed. Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ---------------------------------------------------------
    # 6. Checkpointing
    # ---------------------------------------------------------
    print("\n[6] Saving Checkpoint...")
    # Save the model state. The inference script expects the naming convention: "{model_name}_fold_{fold}.pth"
    checkpoint_filename = f"{Config.model_names[0]}_fold_0.pth"

    save_checkpoint(
        state={"state_dict": model.state_dict()},
        is_best=True,
        output_dir=Config.output_dir,
        filename=checkpoint_filename,
    )

    expected_ckpt_path = os.path.join(Config.output_dir, checkpoint_filename)
    assert os.path.exists(
        expected_ckpt_path
    ), f"Checkpoint not found at {expected_ckpt_path}"
    print(f"Checkpoint successfully saved to {expected_ckpt_path}")

    # ---------------------------------------------------------
    # 7. Inference and Submission
    # ---------------------------------------------------------
    print("\n[7] Running Inference...")

    # Verify Test Loader
    test_loader = get_test_loader(
        image_size=Config.p2_image_size, batch_size=Config.p1_batch_size
    )
    print(f"Test Loader batches: {len(test_loader)}")

    # Run the full inference pipeline provided in library/inference.py
    # This will:
    # 1. Look for checkpoints in Config.output_dir matching Config.model_names
    # 2. Run TTA (Test Time Augmentation)
    # 3. Aggregate predictions and save to ./submission/submission.csv
    try:
        run_inference()
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    # Validate Submission File
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file generated. Shape: {df_sub.shape}")
    print("Head of submission:")
    print(df_sub.head())

    # Final Assertions
    assert len(df_sub) == 2676, f"Expected 2676 rows in submission, got {len(df_sub)}"
    assert list(df_sub.columns) == ["image_id", "label"], "Incorrect submission columns"
    assert df_sub["label"].dtype == np.int64, "Label column should be integers"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
