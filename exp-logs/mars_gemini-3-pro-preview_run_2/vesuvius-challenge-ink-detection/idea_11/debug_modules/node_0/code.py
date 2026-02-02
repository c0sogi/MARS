import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encoding, fbeta_score
from library.dataset import prepare_volumes, InkDataset
from library.model import SegFormer
from library.losses import DiceBCELoss
from library.trainer import Trainer
from library.inference import InferenceEngine, z_scan_predict


def run_demo():
    print("=== Starting Vesuvius Ink Detection Demo ===\n")

    # 1. Configuration Overrides for Demo Speed
    print("--- Configuring Environment ---")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./submission.csv"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set Seed
    seed_everything(Config.SEED)
    print(
        f"Configuration set: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Verify Utility Functions
    print("\n--- Verifying Utilities ---")

    # Test RLE
    dummy_mask = np.array([0, 1, 1, 1, 0, 1, 0])
    encoded = rle_encoding(dummy_mask)
    expected_rle = "2 3 6 1"  # 1-based indexing: start at 2 len 3, start at 6 len 1
    assert (
        encoded == expected_rle
    ), f"RLE failed. Expected '{expected_rle}', got '{encoded}'"
    print("RLE Encoding: Verified")

    # Test F-Beta
    # Case 1: Perfect match
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Logits or probs after threshold
    score = fbeta_score(y_pred, y_true, beta=0.5, threshold=0.5)
    assert np.isclose(score, 1.0), f"F-Beta perfect match failed. Got {score}"

    # Case 2: No overlap
    y_pred_bad = torch.tensor([0.0, 1.0, 0.0, 1.0])
    score_bad = fbeta_score(y_pred_bad, y_true, beta=0.5, threshold=0.5)
    assert np.isclose(score_bad, 0.0), f"F-Beta no match failed. Got {score_bad}"
    print("F-Beta Score: Verified")

    # 3. Data Loading & Dataset
    print("\n--- Setting up Data & Volumes ---")

    # Load metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Metadata not found at {train_csv_path}")

    df_full = pd.read_csv(train_csv_path)

    # Subset for speed: Take top 8 samples (enough for 2 batches)
    df_subset = df_full.head(8).copy()
    print(f"Subsetted training data to {len(df_subset)} samples.")

    # Identify fragments in subset
    frag_ids = df_subset["fragment_id"].unique().astype(str).tolist()
    print(f"Loading volumes for fragments: {frag_ids}")

    # Prepare volumes (this loads/caches the 3D data)
    volumes = prepare_volumes(frag_ids, load_cached_data=True)

    # Check volume shape
    for fid, vol in volumes.items():
        print(f"Fragment {fid} volume shape: {vol.shape}")
        # Expecting (Depth, Height, Width). Depth should be CACHE_Z_MAX - CACHE_Z_MIN = 55 - 15 = 40
        assert (
            vol.shape[0] == 40
        ), f"Volume depth mismatch. Expected 40, got {vol.shape[0]}"

    # Instantiate Dataset
    train_ds = InkDataset(df_subset, volumes, z_start=Config.Z_START, mode="train")

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 0 for simple debugging
    )

    # Fetch one batch
    images, labels, masks, indices = next(iter(train_loader))
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Verify shapes: (B, 3, 512, 512) and (B, 1, 512, 512)
    assert images.shape == (Config.BATCH_SIZE, 3, Config.TILE_SIZE, Config.TILE_SIZE)
    assert labels.shape == (Config.BATCH_SIZE, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    print("Dataset & DataLoader: Verified")

    # 4. Model & Loss
    print("\n--- Initializing Model & Loss ---")
    model = SegFormer().to(Config.DEVICE)
    criterion = DiceBCELoss()

    # Forward pass check
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    logits = model(images)
    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == labels.shape, "Model output shape mismatch"

    loss = criterion(logits, labels)
    print(f"Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model & Loss: Verified")

    # 5. Training Loop
    print("\n--- Running Trainer (1 Epoch) ---")

    # Setup Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Use the same loader for val to save time/memory in demo
    val_loader = train_loader

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # Override baseline score to ensure we save the model for inference testing
    trainer.best_score = -1.0

    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    saved_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(saved_model_path), "Model checkpoint was not saved."
    print("Training Loop: Verified")

    # 6. Inference
    print("\n--- Running Inference ---")

    # Initialize Engine with the model we just trained
    engine = InferenceEngine(model_path=saved_model_path, device=Config.DEVICE)

    # Check Test Metadata
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        test_frags = df_test["fragment_id"].unique()

        if len(test_frags) > 0:
            target_frag = str(test_frags[0])
            print(f"Running prediction on fragment: {target_frag}")

            # Predict
            prob_map = engine.predict_fragment(target_frag, load_cached_data=True)

            print(f"Prediction Map Shape: {prob_map.shape}")
            print(
                f"Prediction Value Range: [{prob_map.min():.4f}, {prob_map.max():.4f}]"
            )

            # Verify shape matches mask
            mask_path = os.path.join(Config.INPUT_DIR, "test", target_frag, "mask.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            assert (
                prob_map.shape == mask.shape
            ), "Prediction shape does not match original mask"

            print("Inference Engine: Verified")
        else:
            print("No test fragments found in metadata.")
    else:
        print("Test metadata file missing.")

    # 7. Full Pipeline Submission Generation
    print("\n--- Generating Submission ---")
    # This runs the full loop over all test fragments and writes submission.csv
    z_scan_predict(load_cached_data=True)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        # Validate content format
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        if not sub_df.empty:
            print("Sample Submission Row:")
            print(sub_df.head(1))
            assert (
                "Id" in sub_df.columns and "Predicted" in sub_df.columns
            ), "Submission columns missing"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
