import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.dataset import GPUResidentDataset
from library.model import EfficientNetV2Audio
from library.engine import Engine
from library.utils import set_seed


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up demo environment in {DEMO_DIR}...")

    # Override Config paths to use the demo directory for caching
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_TRAIN_WAVEFORMS = os.path.join(DEMO_DIR, "train_waveforms.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.CACHE_VAL_WAVEFORMS = os.path.join(DEMO_DIR, "val_waveforms.npy")
    Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.CACHE_TEST_WAVEFORMS = os.path.join(DEMO_DIR, "test_waveforms.npy")
    Config.CACHE_TEST_FNAMES = os.path.join(DEMO_DIR, "test_labels.npy")
    Config.CACHE_BACKGROUND_NOISE = os.path.join(DEMO_DIR, "background_noise.npy")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce training parameters for speed
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.PATIENCE = 2

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Create Data Subsets
    # ==========================================
    print("Creating data subsets for rapid execution...")

    # Load original metadata
    df_train_orig = pd.read_csv("./metadata/train.csv")
    df_val_orig = pd.read_csv("./metadata/val.csv")
    df_test_orig = pd.read_csv("./metadata/test.csv")

    # Create Train Subset (ensure mix of commands and background noise)
    df_train_cmds = df_train_orig[~df_train_orig["is_background"]].head(50)
    df_train_bg = df_train_orig[df_train_orig["is_background"]].head(5)
    df_train_sub = pd.concat([df_train_cmds, df_train_bg]).reset_index(drop=True)

    # Create Val Subset
    df_val_sub = df_val_orig.head(20).reset_index(drop=True)

    # Create Test Subset
    df_test_sub = df_test_orig.head(20).reset_index(drop=True)

    # Save subsets
    train_csv_path = os.path.join(DEMO_DIR, "train_subset.csv")
    val_csv_path = os.path.join(DEMO_DIR, "val_subset.csv")
    test_csv_path = os.path.join(DEMO_DIR, "test_subset.csv")

    df_train_sub.to_csv(train_csv_path, index=False)
    df_val_sub.to_csv(val_csv_path, index=False)
    df_test_sub.to_csv(test_csv_path, index=False)

    # Point Config to these new CSVs
    Config.TRAIN_CSV = train_csv_path
    Config.VAL_CSV = val_csv_path
    Config.TEST_CSV = test_csv_path

    print(f"Train subset: {len(df_train_sub)} samples")
    print(f"Val subset: {len(df_val_sub)} samples")
    print(f"Test subset: {len(df_test_sub)} samples")

    # ==========================================
    # 3. Instantiate Datasets
    # ==========================================
    print("\n--- Initializing Datasets ---")

    # Initialize Train Dataset
    # This will trigger loading, processing, and caching of the subset
    train_dataset = GPUResidentDataset(mode="train", load_cached_data=False)

    # Validate Train Dataset
    assert len(train_dataset) == len(
        df_train_sub
    ), f"Train dataset length mismatch: {len(train_dataset)} vs {len(df_train_sub)}"
    assert train_dataset.waveforms.shape == (
        len(df_train_sub),
        Config.NUM_SAMPLES,
    ), "Train waveform tensor shape incorrect"
    assert train_dataset.labels.shape == (
        len(df_train_sub),
    ), "Train labels tensor shape incorrect"
    assert train_dataset.background_noise is not None, "Background noise not loaded"

    print("Train dataset loaded and verified.")

    # Initialize Val Dataset
    val_dataset = GPUResidentDataset(mode="val", load_cached_data=False)
    assert len(val_dataset) == len(df_val_sub)
    print("Val dataset loaded and verified.")

    # Initialize Test Dataset
    test_dataset = GPUResidentDataset(mode="test", load_cached_data=False)
    assert len(test_dataset) == len(df_test_sub)
    print("Test dataset loaded and verified.")

    # ==========================================
    # 4. Model Initialization & Verification
    # ==========================================
    print("\n--- Initializing Model ---")

    # Pass background noise from dataset to model (for augmentation)
    model = EfficientNetV2Audio(background_noise=train_dataset.background_noise)
    model.to(Config.DEVICE)

    # Verify Frontend Buffer
    assert hasattr(
        model.frontend, "background_noise"
    ), "Frontend missing background_noise buffer"
    assert model.frontend.background_noise.device == torch.device(
        Config.DEVICE
    ), "Background noise buffer not on correct device"

    # Run Dummy Forward Pass
    print("Running dummy forward pass...")
    dummy_input = torch.randn(4, Config.NUM_SAMPLES).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        4,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (4, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Training Loop (Engine)
    # ==========================================
    print("\n--- Starting Training Loop ---")

    engine = Engine(model, device=Config.DEVICE)

    # Run training
    # This will run for Config.MAX_EPOCHS (set to 2 above)
    engine.fit(train_dataset, val_dataset)

    # Verify Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training completed and checkpoint saved.")

    # ==========================================
    # 6. Inference
    # ==========================================
    print("\n--- Running Inference ---")

    fnames, predictions = engine.predict(test_dataset)

    # Verify Predictions
    assert len(fnames) == len(
        df_test_sub
    ), "Number of predictions does not match test set size"
    assert len(predictions) == len(
        df_test_sub
    ), "Number of predicted labels does not match test set size"
    assert isinstance(predictions[0], str), "Predicted labels should be strings"
    assert (
        predictions[0] in Config.LABELS
    ), f"Predicted label {predictions[0]} not in valid labels"

    # Create Submission
    submission_df = pd.DataFrame({"fname": fnames, "label": predictions})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
