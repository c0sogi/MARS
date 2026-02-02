import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import MultiStageModel
from library.loss import BoundaryAwareLoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a demo environment with a subset of data to ensure fast execution.
    """
    print(">>> Setting up demo environment...")

    # Define paths
    base_work_dir = "./working/demo_run"
    demo_meta_dir = os.path.join(base_work_dir, "metadata")

    # Clean up previous run if exists
    if os.path.exists(base_work_dir):
        shutil.rmtree(base_work_dir)

    os.makedirs(demo_meta_dir, exist_ok=True)

    # Create subset metadata (Top N samples) to speed up data loading
    # We read the original metadata and save a small slice to the demo folder
    original_meta_dir = "./metadata"

    subsets = {"train.csv": 20, "val.csv": 10, "test.csv": 10}

    for filename, count in subsets.items():
        src = os.path.join(original_meta_dir, filename)
        dst = os.path.join(demo_meta_dir, filename)

        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take a subset
            df_subset = df.head(count)
            df_subset.to_csv(dst, index=False)
            print(f"    Created subset for {filename}: {len(df_subset)} samples")
        else:
            raise FileNotFoundError(f"Original metadata file not found: {src}")

    return base_work_dir, demo_meta_dir


def configure_settings(work_dir, meta_dir):
    """
    Overrides Config attributes for the demo run.
    """
    print(">>> Configuring settings...")

    # Override Global Paths
    Config.METADATA_DIR = meta_dir
    Config.WORK_DIR = work_dir
    Config.CACHE_DIR = os.path.join(work_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(work_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(work_dir, "submission")

    # Override Hyperparameters for Speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 2

    # Initialize directories based on new config
    Config.init_dirs()

    # Set seed for reproducibility
    set_seed(Config.SEED)


def verify_data_loading():
    """
    Verifies that data loaders function correctly and produce expected shapes.
    """
    print(">>> Verifying Data Loading...")

    # Force reload from raw files (load_cached_data=False) to test processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    features = batch["features"]
    labels_cls = batch["labels_cls"]
    labels_bnd = batch["labels_bnd"]
    mask = batch["mask"]

    print(f"    Batch Size: {features.shape[0]}")
    print(f"    Sequence Length: {features.shape[1]}")
    print(f"    Input Dimension: {features.shape[2]}")

    # Assertions
    assert features.dim() == 3, "Features should be (B, T, D)"
    assert labels_cls.dim() == 2, "Class labels should be (B, T)"
    assert labels_bnd.dim() == 2, "Boundary labels should be (B, T)"
    assert mask.dim() == 2, "Mask should be (B, T)"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {features.shape[2]}"

    print("    Data Loading Verified.")
    return train_loader, val_loader, test_loader, batch


def verify_model_and_loss(batch):
    """
    Verifies model forward pass and loss computation.
    """
    print(">>> Verifying Model and Loss...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = MultiStageModel().to(device)

    # Move batch to device
    features = batch["features"].to(device)
    mask = batch["mask"].to(device)
    labels_cls = batch["labels_cls"].to(device)
    labels_bnd = batch["labels_bnd"].to(device)

    # Forward Pass
    outputs = model(features, mask)

    # Verify Outputs
    stages = ["stage1", "stage2", "stage3"]
    for stage in stages:
        assert stage in outputs, f"Output missing {stage}"
        out = outputs[stage]
        cls_logits = out["cls_logits"]
        bnd_logits = out["bnd_logits"]

        # Check shapes: (B, T, C) and (B, T, 1)
        assert cls_logits.shape == (
            features.shape[0],
            features.shape[1],
            Config.NUM_CLASSES,
        )
        assert bnd_logits.shape == (features.shape[0], features.shape[1], 1)

    print("    Model Forward Pass Verified.")

    # Compute Loss
    criterion = BoundaryAwareLoss(device=device)
    loss, metrics = criterion(outputs, labels_cls, labels_bnd, mask)

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert "total_loss" in metrics

    print(f"    Loss Computed: {loss.item():.4f}")
    print("    Loss Calculation Verified.")


def run_training_pipeline():
    """
    Runs the Trainer to demonstrate the full loop: Train -> Val -> Predict.
    """
    print(">>> Running Full Training Pipeline...")

    # Initialize Trainer
    # Note: Trainer calls get_dataloaders internally. Since we updated Config and
    # cleared the cache dir in setup, it will use the subset data.
    trainer = Trainer(load_cached_data=True)

    # Train
    # We set Config.NUM_EPOCHS=2 earlier
    trainer.train()

    # Predict
    trainer.predict()

    # Verify Outputs
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    assert os.path.exists(submission_path), "Submission file not created"
    assert os.path.exists(checkpoint_path), "Model checkpoint not created"

    # Check submission content
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"    Submission generated with {len(lines)} lines.")
        # We expect 10 lines for the 10 test samples in our subset
        assert len(lines) == 10, f"Expected 10 predictions, got {len(lines)}"

    print("    Pipeline Execution Verified.")


if __name__ == "__main__":
    # 1. Setup Demo Environment (Subset Data)
    work_dir, meta_dir = setup_demo_environment()

    # 2. Configure Config to use Demo Environment
    configure_settings(work_dir, meta_dir)

    # 3. Verify Data Loading
    train_loader, val_loader, test_loader, sample_batch = verify_data_loading()

    # 4. Verify Model & Loss Logic
    verify_model_and_loss(sample_batch)

    # 5. Run Trainer (Train/Val/Predict)
    run_training_pipeline()

    print("\n>>> Demo Completed Successfully.")
