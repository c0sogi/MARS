import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, get_taxonomy_mapping
from library.model import HerbariumNet
from library.loss import TaxonomicFocalLoss
from library.train import fit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set up a specific working directory for this demo
    demo_working_dir = "./working/demo_script_output"
    Config.WORKING_DIR = demo_working_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    Config.TAXONOMY_MAP_PATH = os.path.join(
        demo_working_dir, "taxonomy_mapping.parquet"
    )
    Config.create_directories()

    # Enable Debug Mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for quick execution
    Config.NUM_WORKERS = 2  # Reduced workers for small batch

    # Configure Phase 1 for a single quick epoch
    Config.PHASE1["epochs"] = 1
    Config.PHASE1["batch_size"] = 8
    Config.PHASE1["img_size"] = 224

    # Check for existing taxonomy mapping in root working dir to speed up init
    # The provided file list shows one at ./working/taxonomy_mapping.parquet
    existing_map = "./working/taxonomy_mapping.parquet"
    if os.path.exists(existing_map):
        print(
            f"    Copying existing taxonomy map from {existing_map} to save generation time."
        )
        shutil.copy(existing_map, Config.TAXONOMY_MAP_PATH)

    # Initialize Logger and Seed
    logger = get_logger(
        "demo_script", log_file=os.path.join(demo_working_dir, "run.log")
    )
    seed_everything(Config.SEED)

    print("    Configuration complete.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Ensure taxonomy mapping is available (loads from cache or generates)
    # This is required by the Loss function later
    print("    Loading/Generating Taxonomy Mapping...")
    taxonomy_df = get_taxonomy_mapping(load_cached_data=True)
    assert isinstance(
        taxonomy_df, pd.DataFrame
    ), "Taxonomy mapping should be a DataFrame"
    assert (
        len(taxonomy_df) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, got {len(taxonomy_df)}"
    print(f"    Taxonomy mapping loaded with {len(taxonomy_df)} classes.")

    # Get DataLoaders
    print("    Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        img_size=Config.PHASE1["img_size"],
        batch_size=Config.PHASE1["batch_size"],
        debug=Config.DEBUG,
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"    Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert labels.dim() == 1, "Labels should be 1D tensor (B,)"
    assert images.shape[0] == Config.PHASE1["batch_size"], "Batch size mismatch"
    assert images.shape[2] == Config.PHASE1["img_size"], "Image height mismatch"

    print("    Data Loading verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Initialization and Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    # Initialize Model
    # pretrained=False to avoid downloading weights during this timed run
    model = HerbariumNet(pretrained=False)
    model.to(device)

    # Verify Forward Pass
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass with labels (Training mode logic)
    logits = model(images, labels)
    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.PHASE1["batch_size"],
        Config.NUM_CLASSES,
    ), f"Logits shape mismatch. Expected {(Config.PHASE1['batch_size'], Config.NUM_CLASSES)}"

    print("    Model verification successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Taxonomic Focal Loss...")

    criterion = TaxonomicFocalLoss(
        gamma=Config.FOCAL_LOSS_GAMMA, epsilon=Config.LABEL_SMOOTHING_EPS
    )
    criterion.to(device)

    loss = criterion(logits, labels)
    print(f"    Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("    Loss function verification successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (One Epoch)...")

    # We use the 'fit' function from library.train which handles the loop, validation, and saving
    best_score = fit(
        phase_config=Config.PHASE1,
        model=model,
        criterion=criterion,
        device=device,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        start_epoch=0,
        resume_best_score=0.0,
    )

    print(f"    Training complete. Best Validation F1 Score: {best_score:.4f}")

    # Verify Output
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR, f"checkpoint_{Config.PHASE1['name']}.pth"
    )
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"

    # Verify Best Model
    if best_score > 0:
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        # Note: In a very short debug run with random weights, val score might be 0.0,
        # so best_model might not be saved if logic requires strict improvement > 0.
        # However, fit() saves checkpoint_phase1.pth regardless.
        if os.path.exists(best_model_path):
            print("    Best model saved successfully.")
        else:
            print(
                "    Best model not saved (score might be 0.0), but checkpoint exists."
            )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
