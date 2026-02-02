import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import NarrowSEResNet
from library.engine import train_model, predict


def run_cactus_classification_demo():
    # =========================================================================
    # 1. Configuration Setup for Demo
    # =========================================================================
    print("Setting up configuration...")

    # Override Config for speed and specific output location
    Config.EPOCHS = 2  # Reduce epochs for quick demonstration
    Config.SEEDS = [42]  # Run only one seed
    Config.BATCH_SIZE = 32  # Moderate batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set global seed
    seed_everything(42)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing DataLoaders...")

    # load_cached_data=False forces the dataset to read images from disk
    # and verify the pipeline, rather than loading potentially stale .npy files.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Validation: Check batch shapes
    dummy_imgs, dummy_lbls = next(iter(train_loader))
    assert dummy_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Expected train batch shape {(Config.BATCH_SIZE, 3, 32, 32)}, got {dummy_imgs.shape}"
    assert dummy_lbls.shape == (
        Config.BATCH_SIZE,
    ), f"Expected train label shape {(Config.BATCH_SIZE,)}, got {dummy_lbls.shape}"

    print(f"Data loaded successfully. Train batches: {len(train_loader)}")

    # =========================================================================
    # 3. Training Loop
    # =========================================================================
    test_preds_accumulator = []

    for seed in Config.SEEDS:
        print(f"\n--- Running for Seed {seed} ---")
        seed_everything(seed)

        # Initialize Model
        model = NarrowSEResNet().to(device)

        # Validation: Check model output shape
        with torch.no_grad():
            dummy_out = model(dummy_imgs.to(device))
            assert dummy_out.shape == (
                Config.BATCH_SIZE,
                1,
            ), f"Expected model output shape {(Config.BATCH_SIZE, 1)}, got {dummy_out.shape}"

        # Setup Optimizer and Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Train
        best_auc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            seed=seed,
        )

        # =========================================================================
        # 4. Inference
        # =========================================================================
        print(f"Loading best model for Seed {seed} and running inference...")

        # Load best checkpoint
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Predict
        ids, preds = predict(model, test_loader, device)
        test_preds_accumulator.append(preds)

        # Validation: Check predictions
        assert len(ids) == len(preds), "Mismatch between IDs and predictions count."
        assert np.all(
            (preds >= 0) & (preds <= 1)
        ), "Predictions out of probability range [0, 1]."

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    print("\nGenerating submission file...")

    # Average predictions across seeds (if multiple seeds were used)
    avg_preds = np.mean(test_preds_accumulator, axis=0)

    submission_df = pd.DataFrame({"id": ids, "has_cactus": avg_preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    # Final Validation of Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_check.shape == (
        len(ids),
        2,
    ), f"Submission shape mismatch. Expected {(len(ids), 2)}, got {df_check.shape}"
    assert list(df_check.columns) == [
        "id",
        "has_cactus",
    ], "Incorrect columns in submission file."

    print("Demo completed successfully.")


if __name__ == "__main__":
    run_cactus_classification_demo()
