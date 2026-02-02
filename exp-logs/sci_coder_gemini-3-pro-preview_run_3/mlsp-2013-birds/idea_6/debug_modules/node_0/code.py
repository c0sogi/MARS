import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, get_test_loader
from library.model import BirdResNetSPP
from library.trainer import Trainer


def run_demo():
    print("==== Bird Species Classification Demo ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring parameters for fast demonstration...")
    # Override Config class attributes directly to optimize for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size for demonstration
    Config.N_FOLDS = 2  # Minimum folds required for CV logic
    Config.PRETRAINED = False  # Skip downloading ImageNet weights for speed

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    print("Configuration updated: EPOCHS=1, BATCH_SIZE=8, PRETRAINED=False")

    # -------------------------------------------------------------------------
    # 2. Data Preparation and Loading
    # -------------------------------------------------------------------------
    print("\n[Step 2] Preparing DataLoaders (and caching spectrograms)...")

    # get_dataloaders handles caching internally.
    # We use fold_idx=0 for the demo train/val split.
    train_loader, val_loader = get_dataloaders(fold_idx=0, load_cached_data=True)

    # Verify Train Loader
    try:
        images, labels, rec_ids = next(iter(train_loader))
        print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions to verify data pipeline logic
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            224,
            224,
        ), f"Expected image shape ({Config.BATCH_SIZE}, 3, 224, 224), got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
            19,
        ), f"Expected label shape ({Config.BATCH_SIZE}, 19), got {labels.shape}"
        assert isinstance(rec_ids, torch.Tensor), "rec_ids should be a Tensor"
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Verify Val Loader
    try:
        val_images, val_labels, _ = next(iter(val_loader))
        print(f"Val Batch   - Images: {val_images.shape}, Labels: {val_labels.shape}")
    except StopIteration:
        raise RuntimeError("Val loader is empty!")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing BirdResNetSPP model...")
    device = get_device()
    print(f"Device: {device}")

    model = BirdResNetSPP(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Verify Forward Pass with a dummy batch
    with torch.no_grad():
        dummy_input = images.to(device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (Config.BATCH_SIZE, 19), "Model output shape mismatch!"

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    trainer = Trainer(model, device=device)

    # Run fit for Fold 0
    # This will train for 1 epoch (Config.EPOCHS) and save the best model
    best_auc = trainer.fit(train_loader, val_loader, fold_idx=0)

    print(f"Training finished. Best AUC: {best_auc:.4f}")

    # Verify Checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fold_0_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint verified at: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference on Test Set
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Set...")
    test_loader = get_test_loader(load_cached_data=True)

    # Predict
    rec_ids_test, probs_test = trainer.predict(test_loader)

    print(f"Prediction complete. Count: {len(rec_ids_test)}")
    print(f"Probability Matrix Shape: {probs_test.shape}")

    # The test set has 64 samples (from metadata analysis)
    assert (
        len(rec_ids_test) == 64
    ), f"Expected 64 test samples, found {len(rec_ids_test)}"
    assert probs_test.shape == (64, 19), "Probability matrix shape mismatch"

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Submission File...")

    # Task requirement: Id = rec_id * 100 + species_number
    # Submission format: Id, Probability

    submission_data = []
    for i, rid in enumerate(rec_ids_test):
        rid = int(rid)
        for species_idx in range(19):
            row_id = rid * 100 + species_idx
            prob = probs_test[i, species_idx]
            submission_data.append([row_id, prob])

    submission_df = pd.DataFrame(submission_data, columns=["Id", "Probability"])

    # Sort by Id for consistency
    submission_df = submission_df.sort_values("Id")

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Submission Rows: {len(submission_df)}")
    print("Head:")
    print(submission_df.head())

    # Verify File existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
