import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train
import library.predict as predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Logic Verification Script ===")

    # 1. Setup & Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Fast Execution...")
    utils.seed_everything(config.SEED)

    # Reduce image size for faster processing
    config.IMG_HEIGHT = 256
    config.IMG_WIDTH = 256
    config.IMG_SIZE = (256, 256)

    # Reduce batch size and workers
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Use 0 for simple debugging/stability

    # Reduce training duration
    config.EPOCHS = 1

    # Disable downloading pretrained weights to ensure offline execution speed
    config.PRETRAINED = False

    # Define temporary paths for subset metadata
    subset_dir = "./working/debug_metadata"
    os.makedirs(subset_dir, exist_ok=True)

    subset_train_path = os.path.join(subset_dir, "train.csv")
    subset_val_path = os.path.join(subset_dir, "val.csv")
    subset_test_path = os.path.join(subset_dir, "test.csv")

    # 2. Create Data Subsets
    # ---------------------------------------------------------
    print("[2] Creating Metadata Subsets...")

    # Load original metadata
    df_train_orig = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val_orig = pd.read_csv(config.VAL_METADATA_PATH)
    df_test_orig = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subsets (take top N to ensure reproducibility and file existence)
    N_SAMPLES = 16
    df_train_sub = df_train_orig.head(N_SAMPLES).copy()
    df_val_sub = df_val_orig.head(N_SAMPLES).copy()
    df_test_sub = df_test_orig.head(N_SAMPLES).copy()

    # Save subsets
    df_train_sub.to_csv(subset_train_path, index=False)
    df_val_sub.to_csv(subset_val_path, index=False)
    df_test_sub.to_csv(subset_test_path, index=False)

    # Point config to subsets
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path

    print(f"    Subsets created with {N_SAMPLES} samples each.")

    # 3. Verify Metric Logic
    # ---------------------------------------------------------
    print("\n[3] Verifying Metric (pF1 Score)...")

    # Case 1: Perfect prediction
    labels = np.array([0, 1, 0, 1])
    preds_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    score = utils.pf1_score(labels, preds_perfect)
    assert np.isclose(score, 1.0), f"Expected pF1=1.0 for perfect preds, got {score}"

    # Case 2: All wrong (inverted)
    preds_wrong = np.array([1.0, 0.0, 1.0, 0.0])
    score = utils.pf1_score(labels, preds_wrong)
    assert np.isclose(score, 0.0), f"Expected pF1=0.0 for wrong preds, got {score}"

    print("    Metric verification passed.")

    # 4. Verify Data Loading
    # ---------------------------------------------------------
    print("\n[4] Verifying Data Loading...")

    # Force reload to use our new subset CSVs (ignore existing parquet cache)
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=False)

    # Fetch one batch
    images, ages, implants, targets = next(iter(train_loader))

    # Check shapes
    # Images: (B, 1, H, W) -> Channel 1 because we load grayscale
    assert images.shape == (
        config.BATCH_SIZE,
        1,
        config.IMG_HEIGHT,
        config.IMG_WIDTH,
    ), f"Image shape mismatch: {images.shape}"

    # Metadata: (B,)
    assert ages.shape == (config.BATCH_SIZE,), f"Age shape mismatch: {ages.shape}"
    assert implants.shape == (
        config.BATCH_SIZE,
    ), f"Implant shape mismatch: {implants.shape}"

    # Targets: (B,)
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Target shape mismatch: {targets.shape}"

    print("    DataLoader shapes verified.")

    # 5. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")

    net = model.MetadataEfficientNet()
    net.to(config.DEVICE)
    net.eval()

    # Move batch to device
    images = images.to(config.DEVICE)
    ages = ages.to(config.DEVICE)
    implants = implants.to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = net(images, ages, implants)

    # Check output shape: (B, 1) or (B, NUM_CLASSES)
    assert logits.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Model output shape mismatch: {logits.shape}"

    print("    Model forward pass successful.")

    # 6. Run Training Pipeline
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    # We use load_cached_data=True here because get_dataloaders in step 4
    # already generated the parquet files for the subsets.
    train.run_training(epochs=config.EPOCHS, load_cached_data=True)

    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("    Training complete. Model saved.")

    # 7. Run Inference Pipeline
    # ---------------------------------------------------------
    print("\n[7] Running Inference Pipeline...")

    # Run prediction using the model we just trained
    predict.run_prediction(
        model_path=config.MODEL_SAVE_PATH,
        output_path=config.SUBMISSION_PATH,
        load_cached_data=True,
    )

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."
    print("    Inference complete. Submission saved.")

    # 8. Verify Submission Format
    # ---------------------------------------------------------
    print("\n[8] Verifying Submission Format...")

    sub_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["prediction_id", "cancer"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {sub_df.columns}"

    # Check row count
    # Our subset test csv has N_SAMPLES rows (images).
    # The submission is aggregated by prediction_id.
    # In the subset, prediction_ids might be unique or repeated (multiple views).
    # We check that we have at least 1 row and no more than N_SAMPLES.
    assert len(sub_df) > 0, "Submission file is empty."
    assert len(sub_df) <= N_SAMPLES, "Submission has more rows than input images."

    # Check value range
    assert (
        sub_df["cancer"].min() >= 0.0 and sub_df["cancer"].max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print("    Submission format verified.")
    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
