import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_pos_weights, compute_auc
from library.dataset import BirdDataset
from library.transforms import cyclic_roll
from library.models import BirdClassifier
from library.engine import train_one_epoch, evaluate, inference_with_tta


def main():
    print("Starting Library Usage Demonstration...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Small subset for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.TTA_STEPS = 2  # Reduce TTA steps for speed

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading & Dataset Verification
    # =========================================================================
    print("\n--- Testing Data Loading & Dataset ---")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_val_full = pd.read_csv(Config.VAL_CSV)

    # Create subsets for demonstration
    df_train_demo = df_train_full.head(Config.DEBUG_SUBSET_SIZE).copy()
    df_val_demo = df_val_full.head(Config.DEBUG_SUBSET_SIZE).copy()

    print(
        f"Loaded {len(df_train_demo)} training samples and {len(df_val_demo)} validation samples."
    )

    # Instantiate Dataset
    # We use 'resnet18' which implies 224x448 resolution based on Config
    train_dataset = BirdDataset(
        df=df_train_demo,
        phase="train",
        model_name="resnet18",
        load_cached_data=False,  # Force load from disk to test logic
    )

    # Verify Dataset Item Structure
    sample_item = train_dataset[0]

    # Check keys
    required_keys = ["image", "labels", "soft_labels", "rec_id"]
    for key in required_keys:
        assert key in sample_item, f"Missing key in dataset item: {key}"

    # Check Image Shape: (3, H, W) -> Pseudo-RGB
    img_shape = sample_item["image"].shape
    expected_h, expected_w = Config.IMG_SIZE_ANCHOR
    assert img_shape[0] == 3, f"Expected 3 channels, got {img_shape[0]}"
    assert (
        img_shape[1] == expected_h
    ), f"Expected height {expected_h}, got {img_shape[1]}"
    assert (
        img_shape[2] == expected_w
    ), f"Expected width {expected_w}, got {img_shape[2]}"

    # Check Label Shape
    assert (
        sample_item["labels"].shape[0] == Config.NUM_SPECIES
    ), f"Expected {Config.NUM_SPECIES} labels, got {sample_item['labels'].shape[0]}"

    print("Dataset verification passed.")

    # =========================================================================
    # 3. Transform Verification (Cyclic Roll)
    # =========================================================================
    print("\n--- Testing Transforms (Cyclic Roll) ---")

    # Create a dummy image (H, W, C)
    h, w = 100, 100
    dummy_img = np.zeros((h, w, 1), dtype=np.uint8)
    # Draw a vertical line on the left side
    dummy_img[:, 0:10, 0] = 255

    # Apply cyclic roll (shift by 0.5 i.e., 50 pixels)
    rolled_img = cyclic_roll(dummy_img, shift_ratio=0.5)

    # The line should now be in the middle (pixels 50-60)
    # Check the center pixel
    assert rolled_img[50, 55, 0] == 255, "Cyclic roll did not shift features correctly."
    assert rolled_img[50, 5, 0] == 0, "Original location should be empty after shift."

    print("Transform logic verification passed.")

    # =========================================================================
    # 4. Model Verification
    # =========================================================================
    print("\n--- Testing Model Architecture ---")

    # Initialize Model (ResNet18)
    # pretrained=False to avoid downloading weights during this demo run
    model = BirdClassifier(model_name="resnet18", pretrained=False)
    model.to(device)

    # Create dummy batch
    dummy_batch = torch.randn(2, 3, expected_h, expected_w).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_batch)

    # Check Output Shape: (Batch, Num_Species)
    assert logits.shape == (
        2,
        Config.NUM_SPECIES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_SPECIES}), got {logits.shape}"

    print("Model forward pass verification passed.")

    # =========================================================================
    # 5. Training Engine Verification
    # =========================================================================
    print("\n--- Testing Training Loop ---")

    # Prepare DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_dataset = BirdDataset(
        df=df_val_demo, phase="val", model_name="resnet18", load_cached_data=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Optimizer & Loss Weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    label_cols = [c for c in df_train_demo.columns if c.startswith("species_")]
    pos_weights = get_pos_weights(df_train_demo, label_cols, device=device)

    # Run Training for 1 Epoch
    print("Running training epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weights)
    print(f"Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss returned NaN."

    # Run Evaluation
    print("Running evaluation...")
    val_loss, val_auc = evaluate(model, val_loader, device, pos_weights)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss returned NaN."
    assert 0.0 <= val_auc <= 1.0, "AUC score out of bounds."

    print("Training engine verification passed.")

    # =========================================================================
    # 6. Inference & TTA Verification
    # =========================================================================
    print("\n--- Testing Inference with TTA ---")

    # Use validation set as a proxy for test set
    test_df_demo = df_val_demo.copy()

    # Run TTA Inference
    # This uses the 'inference_with_tta' function from engine.py
    preds = inference_with_tta(
        model,
        test_df_demo,
        device,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Check Predictions Shape: (Num_Samples, Num_Species)
    expected_shape = (len(test_df_demo), Config.NUM_SPECIES)
    assert (
        preds.shape == expected_shape
    ), f"Prediction shape mismatch. Expected {expected_shape}, got {preds.shape}"

    # Check Probability Range
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]."

    print("Inference verification passed.")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n--- Generating Sample Submission ---")

    # The submission format requires flattening the predictions.
    # Id = rec_id * 100 + species_id

    submission_rows = []
    rec_ids = test_df_demo["rec_id"].values

    for i, rec_id in enumerate(rec_ids):
        probs = preds[i]
        for species_idx, prob in enumerate(probs):
            submission_id = rec_id * 100 + species_idx
            submission_rows.append({"Id": int(submission_id), "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Save to file
    sub_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    df_submission.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission shape: {df_submission.shape}")

    # Verify submission structure
    assert list(df_submission.columns) == [
        "Id",
        "Probability",
    ], "Incorrect submission columns."
    assert (
        len(df_submission) == len(test_df_demo) * Config.NUM_SPECIES
    ), "Incorrect number of submission rows."

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
