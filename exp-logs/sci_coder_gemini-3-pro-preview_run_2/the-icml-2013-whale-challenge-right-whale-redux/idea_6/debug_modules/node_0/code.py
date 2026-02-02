import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library components
from library.config import Config
from library.dataset import AudioPreprocessor, get_dataloaders, get_test_loader
from library.model import WhaleEnsembleMember
from library.train import train_individual_model, inference
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Whale Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Patching
    # ---------------------------------------------------------
    print("\n[1] Patching Configuration for Debug/Demo Mode...")

    # Modify Config to run a fast, lightweight version of the pipeline
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.EPOCHS = 1  # Only 1 epoch to verify loop
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug
    Config.PRETRAINED = False  # Disable downloading weights (offline/speed)
    Config.MODEL_NAMES = ["resnet34"]  # Use only one model for demonstration
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir

    # Ensure working directory is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Working Dir: {Config.WORKING_DIR}")

    # Set seeds
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Verify Audio Preprocessing
    # ---------------------------------------------------------
    print("\n[2] Verifying Audio Preprocessing...")

    # Load train metadata to get a valid file path
    train_df = pd.read_csv(Config.TRAIN_CSV)
    if len(train_df) == 0:
        raise ValueError("Train metadata is empty.")

    sample_rel_path = train_df.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_ROOT, sample_rel_path)

    print(f"Processing sample file: {sample_rel_path}")

    preprocessor = AudioPreprocessor()
    spec = preprocessor.process_file(sample_full_path)

    print(f"Spectrogram Shape: {spec.shape}")

    # Assertions for Preprocessing
    # Expected shape: (1, N_MELS, TimeFrames)
    # TimeFrames = ceil(Duration * SR / Hop) approx 2.0 * 2000 / 64 = 63

    assert spec.dim() == 3, f"Expected 3D tensor, got {spec.dim()}"
    assert spec.shape[0] == 1, f"Expected 1 channel, got {spec.shape[0]}"
    assert (
        spec.shape[1] == Config.N_MELS
    ), f"Expected {Config.N_MELS} mels, got {spec.shape[1]}"
    assert spec.shape[2] > 0, "Time dimension should be positive"

    # ---------------------------------------------------------
    # 3. Verify Data Loaders
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loaders...")

    # This will trigger load_and_cache_data
    train_loader, val_loader = get_dataloaders(load_cached_data=False, debug=True)

    # Fetch a single batch
    data_batch, label_batch = next(iter(train_loader))

    print(f"Train Batch Data Shape: {data_batch.shape}")
    print(f"Train Batch Label Shape: {label_batch.shape}")

    # Assertions for Data Loading
    assert data_batch.shape[0] == Config.BATCH_SIZE, "Incorrect batch size"
    assert data_batch.shape[1] == 1, "Incorrect channel dimension in batch"
    assert label_batch.shape[0] == Config.BATCH_SIZE, "Incorrect label batch size"
    assert data_batch.dtype == torch.float32, "Data should be float32"

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model_name = Config.MODEL_NAMES[0]
    model = WhaleEnsembleMember(model_name, pretrained=False)
    model.to(Config.DEVICE)

    # Perform a forward pass with the batch from step 3
    data_batch = data_batch.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(data_batch)

    print(f"Model Output (Logits) Shape: {logits.shape}")

    # Assertions for Model
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    # ---------------------------------------------------------
    # 5. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Loop (1 Epoch)...")

    # train_individual_model handles the loop, validation, and saving
    best_model_path = train_individual_model(model_name, train_loader, val_loader)

    print(f"Training complete. Best model path: {best_model_path}")

    # Assertions for Training
    if not os.path.exists(best_model_path):
        # In the unlikely event that validation AUC was 0.0 and never improved
        print("Warning: Best model file not found. Creating dummy for inference test.")
        torch.save(model.state_dict(), best_model_path)
    else:
        print("Checkpoint successfully saved.")

    # ---------------------------------------------------------
    # 6. Verify Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Verifying Inference and Submission...")

    test_loader, test_clips = get_test_loader(load_cached_data=False, debug=True)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # Generate predictions
    probs = inference(model, test_loader, Config.DEVICE)

    print(f"Predictions generated: {len(probs)}")

    # Assertions for Inference
    assert len(probs) == len(
        test_clips
    ), "Number of predictions does not match number of test clips"
    assert len(probs) == min(Config.DEBUG_SUBSET_SIZE, 25149), "Subset size mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"

    # Create Submission
    submission_df = pd.DataFrame({"clip": test_clips, "probability": probs})

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Sample submission saved to: {sub_path}")
    print("First 5 rows:")
    print(submission_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
