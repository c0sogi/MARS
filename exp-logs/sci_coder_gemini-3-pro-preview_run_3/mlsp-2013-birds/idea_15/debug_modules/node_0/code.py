import sys
import os
import shutil
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.models import BirdModel
from library.training import train_fold, predict


def run_demo():
    print("==== Starting Demo Execution ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("1. Configuring environment for demo...")

    # Override Config parameters for a fast, self-contained run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Use a tiny subset of data
    Config.WORKING_DIR = "./working/demo_execution"

    # Update derived paths based on the new working directory
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs to ensure a fresh start
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories and print setup
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n2. Loading Data...")
    # get_dataloaders handles caching (converting BMPs to numpy) and creating loaders
    # debug=True triggers the subsetting logic in get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Verify Train Loader
    try:
        images, labels, rec_ids = next(iter(train_loader))
        print(
            f"   Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, IDs: {rec_ids.shape}"
        )

        # Assertions to ensure data integrity
        assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
        assert images.shape[1] == 3, "Images must have 3 channels (RGB)"
        assert (
            images.shape[2] == Config.IMG_SIZE
        ), f"Image height must be {Config.IMG_SIZE}"
        assert (
            labels.shape[1] == Config.NUM_CLASSES
        ), f"Labels must have {Config.NUM_CLASSES} classes"
        assert len(rec_ids) == Config.BATCH_SIZE, "Batch size mismatch"
        print("   Data Loading Verified.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n3. Initializing Model...")
    # Instantiate the model with a lightweight backbone
    model = BirdModel(backbone_name="resnet18", pretrained=True)
    model.to(Config.DEVICE)

    # Verify Forward Pass
    print("   Verifying forward pass...")
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        outputs = model(dummy_input)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
    print("   Model Initialization Verified.")

    # -------------------------------------------------------------------------
    # 4. Training Simulation
    # -------------------------------------------------------------------------
    print("\n4. Running Training Loop (1 Fold, 1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run training for Fold 0
    trained_model, best_auc = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        fold_idx=0,
        num_epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    print(f"   Training finished. Best AUC: {best_auc:.4f}")

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(
        Config.CHECKPOINT_DIR, "resnet18_fold_0_best.pth"
    )
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n5. Generating Predictions on Test Set...")
    ids, probs = predict(trained_model, test_loader, Config.DEVICE)

    assert len(ids) > 0, "No predictions generated"
    assert len(ids) == len(probs), "Mismatch between IDs and probability vectors"
    print(f"   Generated predictions for {len(ids)} recordings.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n6. Formatting Submission...")

    submission_rows = []
    # Format: Id = rec_id * 100 + species_id
    for i, rec_id in enumerate(ids):
        rec_probs = probs[i]
        for species_idx, prob in enumerate(rec_probs):
            composite_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": composite_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Sort for consistency
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # Save to disk
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"   Submission saved to {Config.SUBMISSION_PATH}")

    # Final Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(saved_df.columns) == [
        "Id",
        "Probability",
    ], "Submission columns are incorrect"
    assert (
        len(saved_df) == len(ids) * Config.NUM_CLASSES
    ), "Incorrect number of rows in submission"

    print("\n==== Demo Execution Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
