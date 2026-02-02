import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import (
    load_and_cache_split,
    CactusDataset,
    MixupCollate,
    get_transforms,
)
from library.models import CactusResNet, FiLMGenerator
from library.engine import train_one_epoch, validate
from library.stacking import StackingEnsemble


def run_demo():
    print(">>> Starting Cactus Identification Library Demo")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Patching
    # -------------------------------------------------------------------------
    print("\n[1] Configuration Setup")

    # Set reproducible seed
    seed_everything(42)

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch the global Config to point to our demo directory and run fast
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Redirect cache paths to avoid conflicts
    Config.CACHE_TRAIN_IMGS = os.path.join(DEMO_DIR, "train_imgs.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.CACHE_TRAIN_FILESIZES = os.path.join(DEMO_DIR, "train_fsizes.npy")
    Config.CACHE_VAL_IMGS = os.path.join(DEMO_DIR, "val_imgs.npy")
    Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.CACHE_VAL_FILESIZES = os.path.join(DEMO_DIR, "val_fsizes.npy")

    # Create necessary subdirectories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {Config.DEVICE}")
    print(f"Working directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[2] Data Loading & Processing")

    # Load training data using the library function
    # This reads metadata/train_metadata.csv and loads images from input/train/
    print("Loading training split...")
    imgs, labels, fsizes, _ = load_and_cache_split(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TRAIN_FILESIZES,
        Config.INPUT_DIR,
        load_cached=False,  # Force processing from scratch
    )

    # Validations
    print(f"Loaded dataset shape: {imgs.shape}")
    assert len(imgs) > 0, "No images loaded"
    assert imgs.shape[1:] == (3, 32, 32), f"Unexpected image shape: {imgs.shape}"
    assert len(imgs) == len(labels) == len(fsizes), "Data arrays length mismatch"

    # Create a small subset for the demo to ensure speed
    subset_size = 64
    indices = np.random.choice(len(imgs), subset_size, replace=False)

    sub_imgs = imgs[indices]
    sub_labels = labels[indices]

    # Normalize file sizes (Z-score)
    fs_mean = fsizes.mean()
    fs_std = fsizes.std() + 1e-8
    sub_fsizes = (fsizes[indices] - fs_mean) / fs_std

    # Instantiate Dataset
    train_dataset = CactusDataset(
        sub_imgs, sub_labels, sub_fsizes, transform=get_transforms("train")
    )

    # Instantiate DataLoader with MixupCollate
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MixupCollate(alpha=0.2),
        num_workers=0,
    )

    # Verify Batch Generation
    batch_imgs, batch_labels, batch_fs = next(iter(train_loader))
    print(
        f"Batch shapes - Img: {batch_imgs.shape}, Label: {batch_labels.shape}, FSize: {batch_fs.shape}"
    )

    assert batch_imgs.shape == (Config.BATCH_SIZE, 3, 32, 32)
    assert batch_labels.shape == (Config.BATCH_SIZE, 1)
    assert batch_fs.shape == (Config.BATCH_SIZE, 1)
    # Check that data is float tensors
    assert batch_imgs.dtype == torch.float32
    assert batch_labels.dtype == torch.float32

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Model Logic Verification")

    # Instantiate one of the architectures (ResNet with FiLM)
    model = CactusResNet(num_classes=1).to(Config.DEVICE)

    # Move batch to device
    b_imgs = batch_imgs.to(Config.DEVICE)
    b_fs = batch_fs.to(Config.DEVICE)

    # Test FiLM Generator separately (internal component check)
    print("Verifying FiLM Generator...")
    film_gen = FiLMGenerator(num_features=64).to(Config.DEVICE)
    gamma, beta = film_gen(b_fs)
    assert gamma.shape == (Config.BATCH_SIZE, 64, 1, 1)
    assert beta.shape == (Config.BATCH_SIZE, 64, 1, 1)

    # Test Full Model Forward Pass
    print("Verifying Full Model Forward Pass...")
    outputs = model(b_imgs, b_fs)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 1)
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    # -------------------------------------------------------------------------
    # 4. Training Engine (Optimization Loop)
    # -------------------------------------------------------------------------
    print("\n[4] Training Engine Execution")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, torch.device(Config.DEVICE), epoch=0
    )
    print(f"Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Run validation (using the same loader for simplicity in demo)
    # Note: validate() expects raw items, not Mixup, so we create a plain loader
    val_dataset = CactusDataset(
        sub_imgs, sub_labels, sub_fsizes, transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Running validate...")
    val_loss, val_auc = validate(model, val_loader, torch.device(Config.DEVICE))
    print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"

    # -------------------------------------------------------------------------
    # 5. Stacking Ensemble Logic
    # -------------------------------------------------------------------------
    print("\n[5] Stacking Ensemble Verification")

    # Simulate OOF predictions
    # 100 samples, 3 base models
    n_samples = 100
    n_models = 3

    # Synthetic ground truth
    y_true = np.random.randint(0, 2, n_samples)

    # Synthetic predictions (random probabilities)
    X_oof = np.random.rand(n_samples, n_models)

    # Initialize and fit Stacking Ensemble
    stacker = StackingEnsemble(random_state=42)
    stacker.fit(X_oof, y_true)
    assert stacker.is_fitted, "Stacker should be fitted after fit()"

    # Predict
    final_preds = stacker.predict(X_oof)
    assert final_preds.shape == (n_samples,)
    assert (final_preds >= 0).all() and (final_preds <= 1).all()

    # Evaluate
    auc_score = stacker.evaluate(X_oof, y_true)
    print(f"Stacking Ensemble AUC (Synthetic): {auc_score:.4f}")

    print("\n>>> Demo completed successfully. All components verified.")


if __name__ == "__main__":
    run_demo()
